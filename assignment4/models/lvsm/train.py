"""Mini-LVSM training.

    python -m models.lvsm.train --config configs/lvsm.json
    python -m models.lvsm.train --overfit        # sanity check: memorize one batch
"""

# ruff: noqa: F841

import argparse
import json
import math
import time
from pathlib import Path

import torch

from a4data.dataset import lvsm_inputs, lvsm_targets
from models.lvsm.eval import evaluate
from models.lvsm.losses import compute_loss  # noqa: F401 -- used in TODOs
from models.lvsm.metrics import psnr
from models.lvsm.utils import (
    build_model,
    load_config,
    make_loader,
    select_device,
    to_device,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    num_source_views,
    scheduler=None,
    grad_clip=None,
    amp=False,
):
    """Q2.0: Train target RGB synthesis and return a dict of epoch-average losses.

    Args:
        model: MiniLVSM on device.
        loader: make_dataloader(..., split="train"), yielding channel-first batches.
        optimizer: Optimizer for model parameters.
        device: Torch device for model inputs and supervision.
        num_source_views: S, with 1 <= S < V; target count T = V - S.
        scheduler: Optional per-step LR scheduler.
        grad_clip: Optional max gradient norm.
        amp: Use bfloat16 autocast (CUDA only).

    Returns:
        Dict with loss (MSE) and psnr, averaged over every target image seen.
    """
    model.train()
    totals = {"loss": 0.0, "psnr": 0.0}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        inputs = lvsm_inputs(batch, num_source_views)
        targets = lvsm_targets(batch, num_source_views)

        # TODO (Q2.0): Run model(inputs) under torch.autocast with
        # device.type, bfloat16 and enabled=amp.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            predictions = model(inputs)
        # TODO (Q2.0): Cast every prediction to float32 (the loss is computed
        # outside autocast), then compute the loss with compute_loss.
        predictions = {k: v.float() for k, v in predictions.items()}
        loss = compute_loss(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = targets["rgb"].shape[0] * targets["rgb"].shape[1]  # images = B * T
        totals["loss"] += loss.item() * n
        totals["psnr"] += psnr(
            predictions["rgb"].detach().flatten(0, 1), targets["rgb"].flatten(0, 1)
        ).sum().item()
        count += n
    if count == 0:
        raise ValueError("Training loader produced no batches")
    return {k: v / count for k, v in totals.items()}


def make_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup followed by cosine decay to 10% of the peak LR."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer, config, epoch, metrics, best_psnr):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "val": metrics,
            "best_psnr": best_psnr,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/lvsm.json")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--resume", help="Checkpoint to continue from")
    parser.add_argument(
        "--overfit",
        type=int,
        nargs="?",
        const=200,
        metavar="STEPS",
        help="Train STEPS (default 200) times on one fixed batch to check the "
        "implementation; the loss and PSNR should reach ~30+ dB",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tr = config["training"]
    s = config["num_source_views"]
    output_dir = Path(args.output_dir or tr["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.manual_seed(config["data"].get("seed", 0))

    train_loader = make_loader(config, "train")
    if args.overfit:
        batch = next(iter(train_loader))
        train_loader, val_loader = [batch] * args.overfit, [batch]
        epochs, vis_every = 1, 1
    else:
        val_loader = make_loader(config, "val")
        epochs, vis_every = args.epochs or tr["epochs"], tr.get("vis_every", 1)

    model = build_model(config).to(device)
    train_fn, loss_fn = train_one_epoch, compute_loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        betas=tuple(tr.get("betas", (0.9, 0.95))),
    )
    total_steps = epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, tr.get("warmup_steps", 200))
    start_epoch, best_psnr = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", ckpt["val"]["psnr"])
        # Older checkpoints keep the historical best only in the sibling file.
        previous_best = Path(args.resume).parent / "best.pt"
        if "best_psnr" not in ckpt and previous_best.exists():
            best = torch.load(previous_best, map_location="cpu", weights_only=False)
            best_psnr = max(best_psnr, best["val"]["psnr"])
            del best
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Mini-LVSM: {n_params:.1f}M params, device {device}, S={s}, "
          f"{len(train_loader)} steps/epoch, {epochs} epochs")
    amp = tr.get("amp", False) and device.type == "cuda"
    log_path = output_dir / "log.jsonl"
    for epoch in range(start_epoch, epochs):
        start = time.time()
        train = train_fn(
            model, train_loader, optimizer, device, s, scheduler,
            grad_clip=tr.get("grad_clip"), amp=amp,
        )
        vis = output_dir / f"val_epoch{epoch:03d}.png" if epoch % vis_every == 0 else None
        val = evaluate(model, val_loader, device, s, vis_path=vis, loss_fn=loss_fn)
        record = {"epoch": epoch, "train": train, "val": val, "seconds": time.time() - start}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:3d} | train loss {train['loss']:.5f} psnr {train['psnr']:.2f} "
            f"| val loss {val['loss']:.5f} psnr {val['psnr']:.2f} ssim {val['ssim']:.4f} "
            f"| {record['seconds']:.0f}s"
        )
        improved = val["psnr"] > best_psnr
        if improved:
            best_psnr = val["psnr"]
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, config, epoch, val, best_psnr
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, config, epoch, val, best_psnr
            )


if __name__ == "__main__":
    main()
