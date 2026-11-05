"""Mini-LVSM validation/test: PSNR / SSIM / loss, plus the required comparisons.

Examples (run from the repository root):

    # Standard evaluation on the validation split
    python -m models.lvsm.eval --checkpoint outputs/lvsm/best.pt --split val

    # Source-view comparison: 1, 2 and 4 source views
    python -m models.lvsm.eval --checkpoint outputs/lvsm/best.pt --num-source-views 1 2 4

    # Pose-conditioning comparison: correct / shuffled / removed target rays
    python -m models.lvsm.eval --checkpoint outputs/lvsm/best.pt \
        --conditioning correct shuffled none
"""

import argparse
import json
from pathlib import Path

import torch

from a4data.dataset import lvsm_inputs, lvsm_targets
from models.lvsm.losses import compute_loss
from models.lvsm.metrics import psnr, ssim
from models.lvsm.utils import (
    build_model,
    load_config,
    make_loader,
    save_visualization,
    select_device,
    to_device,
)

CONDITIONING_MODES = ("correct", "shuffled", "none")


def apply_conditioning(inputs, mode):
    """Ablate the target-camera conditioning (never touches target RGB).

    correct:  the true target rays.
    shuffled: rays of a *different* target view (rolled over the target axis, or
              over the batch when there is a single target), so the query asks for
              the wrong viewpoint while the sources stay correct.
    none:     all-zero target rays; the query carries no pose information.
    """
    if mode == "correct":
        return inputs
    inputs = dict(inputs)
    rays = inputs["target_plucker"]
    if mode == "none":
        inputs["target_plucker"] = torch.zeros_like(rays)
    elif mode == "shuffled":
        if rays.shape[1] > 1:
            inputs["target_plucker"] = rays.roll(1, dims=1)
        elif rays.shape[0] > 1:
            inputs["target_plucker"] = rays.roll(1, dims=0)
        else:
            raise ValueError("Need >1 target view or >1 sample to shuffle rays")
    else:
        raise ValueError(f"Unknown conditioning mode {mode!r}; use {CONDITIONING_MODES}")
    return inputs


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_source_views,
    conditioning="correct",
    vis_path=None,
    loss_fn=compute_loss,
):
    """Evaluate target RGB synthesis and return aggregated losses/metrics.

    Args:
        model: MiniLVSM on device; call model.eval() before inference.
        loader: make_dataloader(..., split="val" or "test"), or any iterable of
            collated batches.
        device: Torch device for inputs and targets.
        num_source_views: S, using the same partition for inputs and targets.
        conditioning: one of CONDITIONING_MODES (pose-conditioning ablation).
        vis_path: optional PNG path for a picture of the first batch.
        loss_fn: Loss used for the reported `loss` (default: your compute_loss).

    Returns:
        Dict with loss (MSE), psnr (dB), ssim and num_images, each averaged over
        every evaluated target image (so short last batches are weighted correctly).
    """
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        inputs = lvsm_inputs(batch, num_source_views)
        targets = lvsm_targets(batch, num_source_views)
        predictions = model(apply_conditioning(inputs, conditioning))
        pred, gt = predictions["rgb"], targets["rgb"]
        n = gt.shape[0] * gt.shape[1]  # images = B * T
        flat_pred, flat_gt = pred.flatten(0, 1), gt.flatten(0, 1)
        totals["loss"] += loss_fn(predictions, targets).item() * n
        totals["psnr"] += psnr(flat_pred, flat_gt).sum().item()
        totals["ssim"] += ssim(flat_pred, flat_gt).sum().item()
        count += n
        if i == 0 and vis_path is not None:
            save_visualization(vis_path, inputs, targets, predictions)
    if count == 0:
        raise ValueError("Evaluation loader produced no batches")
    result = {k: v / count for k, v in totals.items()}
    result["num_images"] = count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Override the config stored in the checkpoint")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--num-source-views",
        type=int,
        nargs="+",
        help="One or more S values (default: the training value)",
    )
    parser.add_argument(
        "--conditioning", nargs="+", default=["correct"], choices=CONDITIONING_MODES
    )
    parser.add_argument("--device")
    parser.add_argument("--output", help="Directory for metrics.json and images")
    args = parser.parse_args()

    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    loss_fn = compute_loss
    source_counts = args.num_source_views or [config["num_source_views"]]
    output = Path(args.output or Path(args.checkpoint).parent / f"eval_{args.split}")

    results = {}
    for s in source_counts:
        # Data must hold at least S + 1 views; keep the trained view count otherwise.
        views = max(config["data"]["num_views"], s + 1)
        loader = make_loader(
            config,
            args.split,
            num_views=views,
            sampling_range=max(config["data"]["sampling_range"], views),
        )
        for mode in args.conditioning:
            key = f"S={s},conditioning={mode}"
            results[key] = evaluate(
                model,
                loader,
                device,
                s,
                conditioning=mode,
                vis_path=output / f"S{s}_{mode}.png",
                loss_fn=loss_fn,
            )
            r = results[key]
            print(
                f"{key}: PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  "
                f"loss {r['loss']:.5f}  ({r['num_images']} images)"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Saved metrics and images to {output}")


if __name__ == "__main__":
    main()
