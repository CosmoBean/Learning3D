"""Train Mini-VGGT normally or overfit a fixed selection of training scenes."""

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, Subset

from a4data.dataset import vggt_targets
from a4data.loading import make_dataloader
from models.vggt.distributed import setup, shard_loader, sum_values, unwrap
from models.vggt.eval import evaluate
from models.vggt.losses import compute_loss
from models.vggt.runtime import (
    add_arguments,
    build_components,
    fixed_training_loader,
    load_config,
    predict,
    to_device,
)
from models.vggt.visualization import save_loss_curve, visualize_predictions


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    weights=None,
    max_steps=None,
    loss_fn=compute_loss,
    log_fn=None,
    log_every=10,
):
    """Train all views; max_steps repeats the fixed loader only in overfit mode."""
    model.train()
    total, count, steps = 0.0, 0, 0
    camera_total, depth_total = 0.0, 0.0
    batches = (
        loader
        if max_steps is None
        else itertools.islice(itertools.cycle(loader), max_steps)
    )
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        predictions = predict(model, batch, device)
        loss, components = loss_fn(
            predictions,
            to_device(
                vggt_targets(batch, include_points="points_world" in predictions),
                device,
            ),
            weights,
            return_components=True,
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        n = batch["rgb"].shape[0]
        total += loss.detach().item() * n
        count += n
        camera_total += components["camera"].detach().item() * n
        depth_total += components["depth"].detach().item() * n
        steps += 1
        if log_fn is not None and steps % log_every == 0:
            (
                batch_loss,
                batch_count,
                global_total,
                global_count,
                camera_loss,
                depth_loss,
            ) = sum_values(
                [
                    loss.detach().item() * n,
                    n,
                    total,
                    count,
                    components["camera"].detach().item() * n,
                    components["depth"].detach().item() * n,
                ],
                device,
            )
            log_fn(
                {
                    "step": steps,
                    "loss": batch_loss / batch_count,
                    "mean_loss": global_total / global_count,
                    "loss_camera": camera_loss / batch_count,
                    "loss_depth": depth_loss / batch_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
    total, count, camera_total, depth_total = sum_values(
        [total, count, camera_total, depth_total], device
    )
    if not count:
        raise ValueError("Training loader produced no samples")
    return {
        "loss": total / count,
        "loss_camera": camera_total / count,
        "loss_depth": depth_total / count,
        "num_samples": int(count),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument(
        "--overfit", action="store_true", help="Freeze train samples and frame order"
    )
    parser.add_argument("--overfit-scenes", type=int)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        help="Overfit steps; normal training visits every scene once",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument(
        "--init-checkpoint",
        help="Initialize model weights from a checkpoint; start a fresh optimizer/schedule",
    )
    checkpoints.add_argument(
        "--resume", help="Continue a saved run, including its optimizer and schedule"
    )
    parser.add_argument(
        "--log-every", type=int, help="Print and save loss every N training steps"
    )
    parser.add_argument(
        "--max-val-batches", type=int, help="Optional validation limit for smoke tests"
    )
    parser.add_argument("--output", help="Override the run output directory")
    args = parser.parse_args()
    resumed = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    if resumed is not None:
        if any(
            getattr(args, name) is not None
            for name in (
                "config", "epochs", "learning_rate", "overfit_scenes", "steps_per_epoch"
            )
        ):
            parser.error(
                "--resume restores saved training settings; omit training overrides"
            )
        run = resumed["config"]["run"]
        args.overfit = run["overfit"]
        args.overfit_scenes = run["overfit_scenes"]
        args.steps_per_epoch = run["steps_per_epoch"]
        if args.max_val_batches is not None and args.max_val_batches != run["max_val_batches"]:
            parser.error("--resume keeps the original validation batch limit")
        args.max_val_batches = run["max_val_batches"]
    if args.steps_per_epoch is not None and not args.overfit:
        parser.error(
            "--steps-per-epoch is for --overfit; normal mode uses the full train loader"
        )
    config = load_config(args, resumed["config"] if resumed is not None else None)
    settings, data = config["training"], config["data"]
    if resumed is not None and any(
        data[key] != resumed["config"]["data"][key]
        for key in ("num_views", "sampling_range", "image_size", "seed", "batch_size", "eval_num_frames")
        if key in resumed["config"]["data"]
    ):
        parser.error("--resume keeps the original sampling and batch settings")
    default_epochs = (
        settings.get("overfit_epochs", settings["epochs"])
        if args.overfit and resumed is None
        else settings["epochs"]
    )
    epochs = args.epochs if args.epochs is not None else default_epochs
    overfit_count = (
        args.overfit_scenes
        if args.overfit_scenes is not None
        else settings["overfit_scenes"]
    )
    max_steps = (
        (
            args.steps_per_epoch
            if args.steps_per_epoch is not None
            else settings["overfit_steps_per_epoch"]
        )
        if args.overfit
        else None
    )
    if (
        epochs < 1
        or (max_steps is not None and max_steps < 1)
        or (args.max_val_batches is not None and args.max_val_batches < 1)
    ):
        parser.error("Epoch/step/batch limits must be positive")
    settings["epochs"] = epochs
    if args.log_every is not None:
        settings["log_every"] = args.log_every
    if settings.get("log_every", 10) < 1:
        parser.error("--log-every must be positive")
    if args.learning_rate is not None:
        settings["learning_rate"] = args.learning_rate
    elif args.overfit and resumed is None:
        settings["learning_rate"] = settings.get(
            "overfit_learning_rate", settings["learning_rate"]
        )
    config["run"] = {
        **config.get("run", {}),
        "overfit": args.overfit,
        "overfit_scenes": overfit_count,
        "steps_per_epoch": max_steps,
        "max_val_batches": args.max_val_batches,
    }
    rank, world_size, args.device = setup(args.device)
    config["run"]["world_size"] = world_size
    config["run"]["precision"] = (
        "bf16" if torch.device(args.device).type == "cuda" else "fp32"
    )
    torch.manual_seed(data["seed"])
    initial = resumed
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    model, loss_fn, geometry = build_components(
        config, from_checkpoint=initial is not None
    )
    if initial is not None:
        model.load_state_dict(initial["model"])
        if args.init_checkpoint:
            config["run"]["init_checkpoint"] = str(Path(args.init_checkpoint).resolve())
            config["run"]["init_epoch"] = initial["epoch"]
        del initial
    model.to(args.device)
    if args.overfit:
        train_loader, selection = fixed_training_loader(
            data, geometry, overfit_count,
            selection=resumed["overfit_selection"] if resumed is not None else None,
        )
    else:
        train_loader = make_dataloader(data, "train", geometry)
        selection = None
    train_loader = shard_loader(train_loader, data, "train")
    val_loader = (
        shard_loader(make_dataloader(data, "val", geometry), data, "val")
        if not args.overfit
        else None
    )
    overfit_eval_loader = (
        shard_loader(
            fixed_training_loader(data, geometry, selection=selection)[0], data, "val"
        )
        if args.overfit
        else None
    )
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    warmup = 0 if args.overfit else min(settings.get("warmup_epochs", 0), epochs - 1)
    settings["warmup_epochs"] = warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup,
        eta_min=settings["learning_rate"] * settings.get("min_lr_ratio", 0.05),
    )
    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1 / warmup, total_iters=warmup
                ),
                cosine,
            ],
            milestones=[warmup],
        )
        if warmup
        else cosine
    )
    start_epoch, best = 0, float("inf")
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = resumed["epoch"]
        best = resumed.get("best_loss", float("inf"))
        output = Path(args.resume).resolve().parent
        if args.output and Path(args.output).resolve() != output:
            parser.error("--resume continues in the checkpoint directory; omit --output")
    else:
        output = Path(
            args.output or settings["output_dir"] + ("_overfit" if args.overfit else "")
        )
    history = output / "history.jsonl"
    if history.exists() and resumed is None:
        parser.error("Choose a new output directory to preserve the previous run")
    if resumed is not None and history.exists():
        records = [json.loads(line) for line in history.read_text().splitlines()]
        latest_epoch = max((r["epoch"] for r in records), default=0)
        if latest_epoch > start_epoch + 1 or (
            latest_epoch > start_epoch and Path(args.resume).name != "last.pt"
        ):
            parser.error("Checkpoint predates this run's history; resume from last.pt")
        # An interruption can leave an epoch log ahead of the last saved weights.
        records = [r for r in records if r["epoch"] <= start_epoch]
        metric = "overfit_train" if args.overfit else "val"
        for record in records:
            if record["epoch"] > 0:
                best = min(best, record[metric]["loss"])
            if record["epoch"] == start_epoch:
                resumed.setdefault("score", record[metric]["loss"])
        if world_size > 1:
            dist.barrier()  # Every rank must finish reading before history is rewritten.
        if rank == 0:
            history.write_text("".join(json.dumps(r) + "\n" for r in records))
            steps = output / "train_steps.jsonl"
            if steps.exists():
                kept = [
                    line for line in steps.read_text().splitlines()
                    if json.loads(line)["epoch"] <= start_epoch
                ]
                steps.write_text("".join(line + "\n" for line in kept))
        if world_size > 1:
            dist.barrier()
    if resumed is not None:
        # Recover an interruption between saving last.pt and the new best.pt.
        if resumed.get("score") == best:
            if rank == 0:
                torch.save(resumed, output / "best.partial")
                (output / "best.partial").replace(output / "best.pt")
        elif not (output / "best.pt").exists():
            parser.error("Resume needs the original run's best.pt alongside last.pt")
    del resumed
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    settings["output_dir"] = str(output)
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def log(record):
        if rank != 0:
            return
        with history.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        save_loss_curve(history, output / "loss_curve.png")

    def log_step(record):
        if rank != 0:
            return
        record = {"event": "train_step", "epoch": epoch, **record}
        with (output / "train_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    if start_epoch == 0 and args.overfit:
        log(
            {
                "epoch": 0,
                "overfit_train": evaluate(
                    model,
                    overfit_eval_loader,
                    args.device,
                    config["loss"],
                    loss_fn=loss_fn,
                ),
            }
        )
    elif start_epoch == 0:
        log(
            {
                "epoch": 0,
                "val": evaluate(
                    model,
                    val_loader,
                    args.device,
                    config["loss"],
                    args.max_val_batches,
                    loss_fn,
                ),
            }
        )
    for epoch in range(start_epoch + 1, epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        start = time.monotonic()
        train = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            config["loss"],
            max_steps,
            loss_fn,
            log_fn=log_step,
            log_every=settings.get("log_every", 10),
        )
        record = {
            "epoch": epoch,
            "train": train,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        # Only normal training uses held-out validation.
        if not args.overfit:
            record["val"] = evaluate(
                model,
                val_loader,
                args.device,
                config["loss"],
                args.max_val_batches,
                loss_fn,
            )
        if args.overfit:
            record["overfit_train"] = evaluate(
                model, overfit_eval_loader, args.device, config["loss"], loss_fn=loss_fn
            )
        record["seconds"] = round(time.monotonic() - start, 3)
        log(record)
        scheduler.step()
        score = (
            record["overfit_train"]["loss"] if args.overfit else record["val"]["loss"]
        )
        improved = score < best
        best = min(best, score)
        checkpoint = {
            "epoch": epoch,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "overfit_selection": selection,
            "best_loss": best,
            "score": score,
        }
        temporary = output / "last.partial"
        if rank == 0:
            torch.save(checkpoint, temporary)
            temporary.replace(output / "last.pt")
        if improved and rank == 0:
            torch.save(checkpoint, output / "best.partial")
            (output / "best.partial").replace(output / "best.pt")
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        model = unwrap(model)
        checkpoint = torch.load(
            output / "best.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        dataset = train_loader.dataset if args.overfit else val_loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        viewer = visualize_predictions(
            model,
            samples,
            geometry,
            args.device,
            output / "visualization" / "index.html",
        )
        print(f"Saved checkpoints and history to {output.resolve()}")
        print(f"Best-checkpoint GT/prediction viewer: {viewer.resolve()}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
