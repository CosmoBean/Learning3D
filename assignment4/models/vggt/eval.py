"""Evaluate Mini-VGGT checkpoints on validation/test or their fixed overfit samples."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from a4data.dataset import vggt_targets
from a4data.loading import make_dataloader
from models.vggt.distributed import sum_values, unwrap
from models.vggt.losses import compute_loss
from models.vggt.metrics import GeometryMetrics
from models.vggt.runtime import (
    add_arguments,
    build_components,
    fixed_training_loader,
    load_config,
    predict,
    to_device,
)
from models.vggt.visualization import visualize_predictions


@torch.no_grad()
def evaluate(
    model, loader, device, weights=None, max_batches=None, loss_fn=compute_loss
):
    """Return sample-weighted loss plus valid-pixel/pair-weighted metrics."""
    model = unwrap(model)
    model.eval()
    metrics = GeometryMetrics()
    total, count = 0.0, 0
    camera_total, depth_total = 0.0, 0.0
    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for batch in batches:
        predictions = predict(model, batch, device)
        targets = to_device(
            vggt_targets(batch, include_points="points_world" in predictions), device
        )
        n = batch["rgb"].shape[0]
        loss, components = loss_fn(
            predictions, targets, weights, return_components=True
        )
        total += loss.item() * n
        camera_total += components["camera"].item() * n
        depth_total += components["depth"].item() * n
        count += n
        metrics.update(predictions, targets, batch["scene_scale"].to(device))
    metrics.synchronize()
    total, count, camera_total, depth_total = sum_values(
        [total, count, camera_total, depth_total], device
    )
    result = metrics.compute()
    return {
        "loss": total / count,
        "loss_camera": camera_total / count,
        "loss_depth": depth_total / count,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Replay checkpoint's fixed training samples",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--report")
    parser.add_argument("--visualization", help="Write a GT/prediction comparison HTML")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = load_config(args, checkpoint["config"])
    model, loss_fn, geometry = build_components(
        config, from_checkpoint=True
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    if args.overfit:
        if checkpoint["overfit_selection"] is None:
            parser.error("This checkpoint has no fixed overfit samples")
        loader, _ = fixed_training_loader(
            config["data"], geometry, selection=checkpoint["overfit_selection"]
        )
    else:
        loader = make_dataloader(config["data"], args.split, geometry)
    report = {
        "split": "overfit_train" if args.overfit else args.split,
        "checkpoint": args.checkpoint,
        "max_batches": args.max_batches,
        **evaluate(
            model, loader, args.device, config["loss"], args.max_batches, loss_fn
        ),
    }
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.visualization:
        samples = [loader.dataset[i] for i in range(min(4, len(loader.dataset)))]
        path = visualize_predictions(
            model, samples, geometry, args.device, args.visualization
        )
        print(f"GT/prediction viewer: {path.resolve()}")


if __name__ == "__main__":
    main()
