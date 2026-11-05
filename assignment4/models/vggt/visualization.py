"""Compare predicted geometry with GT using the shared standalone HTML viewer."""

import json
from pathlib import Path

import torch
from matplotlib.figure import Figure
from torch.utils.data import default_collate

from a4data.visualization import export_html
from models.vggt.runtime import predict


@torch.no_grad()
def visualize_predictions(model, samples, geometry, device, output):
    """Show RGB, depth, cameras and clouds in the common normalized coordinates.

    samples: list of unbatched dataset samples, each containing V views.
    Predictions receive only RGB; GT validity masks restrict displayed pixels.
    Optional point-head outputs appear as an additional geometry selection.
    """
    model.eval()
    predictions, point_predictions = [], []
    for sample in samples:
        batch = default_collate([sample])
        result = {
            k: v[0]
            .detach()
            .to(device="cpu", dtype=torch.float32 if v.is_floating_point() else v.dtype)
            for k, v in predict(model, batch, device).items()
            if k not in ("pose_enc", "pose_enc_list")
        }
        predicted = dict(sample)
        for key in ("K", "T_world_from_camera", "depth"):
            predicted[key] = result[key]
        predicted["points_world"] = geometry.unproject_depth(
            predicted["K"], predicted["T_world_from_camera"], predicted["depth"]
        ).permute(0, 3, 1, 2)
        predictions.append(predicted)
        if "points_world" in result:
            point_predictions.append(
                {**predicted, "points_world": result["points_world"]}
            )
    return export_html(
        samples,
        output,
        predictions=predictions,
        point_predictions=point_predictions or None,
    )


def save_loss_curve(history, output):
    """Plot logged epoch losses for screenshots during training."""
    records = [json.loads(line) for line in Path(history).read_text().splitlines()]
    series = (
        (("overfit_train", "Train"),)
        if any("overfit_train" in record for record in records)
        else (("train", "Train"), ("val", "Validation"))
    )
    figure = Figure(figsize=(15, 4))
    axes = figure.subplots(1, 3, sharex=True)
    for axis, (component, title) in zip(
        axes,
        (
            ("loss", "Total loss"),
            ("loss_camera", "Camera loss (unweighted)"),
            ("loss_depth", "Depth loss (unweighted)"),
        ),
    ):
        for key, label in series:
            points = [
                (record["epoch"], record[key][component])
                for record in records
                if key in record and component in record[key]
            ]
            if points:
                epochs, losses = zip(*points)
                axis.plot(epochs, losses, label=label, marker="o", markersize=3)
        axis.set(xlabel="Epoch", ylabel="Loss", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output = Path(output)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    return output
