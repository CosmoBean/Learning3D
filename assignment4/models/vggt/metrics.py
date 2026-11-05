"""VGGT-style pairwise pose AUC and masked depth/point metrics.

Reference: facebookresearch/vggt evaluation/test_co3d.py, evaluation branch.
All input poses here are camera-to-world; pair transforms are inv(T_i) @ T_j.
"""

import numpy as np
import torch
import torch.distributed as dist


def relative_pose_errors(predicted, expected):
    """Return rotation/translation-angle errors (B,P), plus defined-baseline mask.

    Translation has the sign ambiguity used in official VGGT evaluation.
    GT baselines <=1e-6 normalized units have no defined direction and are excluded
    from translation and joint AUC metrics. Rotation AUC includes these pairs.
    A zero predicted baseline receives 90 degrees.
    """
    views = expected.shape[1]
    pairs = torch.combinations(torch.arange(views, device=expected.device), r=2)
    i, j = pairs[:, 0], pairs[:, 1]
    predicted, expected = predicted.double(), expected.double()
    relatives = []
    for poses in [predicted, expected]:
        rotations = poses[:, i, :3, :3].transpose(-1, -2)
        relatives.append(
            (
                rotations @ poses[:, j, :3, :3],
                (rotations @ (poses[:, j, :3, 3] - poses[:, i, :3, 3])[..., None])[
                    ..., 0
                ],
            )
        )
    (rp, tp), (rg, tg) = relatives
    difference = rp @ rg.transpose(-1, -2)
    skew = torch.stack(
        [
            difference[..., 2, 1] - difference[..., 1, 2],
            difference[..., 0, 2] - difference[..., 2, 0],
            difference[..., 1, 0] - difference[..., 0, 1],
        ],
        -1,
    )
    sine = skew.norm(dim=-1) / 2
    cosine = (difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    rotation = torch.rad2deg(torch.atan2(sine, cosine.clamp(-1, 1)))
    denominator = tp.norm(dim=-1) * tg.norm(dim=-1)
    cosine_t = ((tp * tg).sum(-1).abs() / denominator.clamp_min(1e-15)).clamp(0, 1)
    translation = torch.rad2deg(torch.acos(cosine_t))
    translation = torch.where(tp.norm(dim=-1) > 1e-6, translation, 90.0)
    return rotation, translation, tg.norm(dim=-1) > 1e-6


def pose_auc(errors, threshold):
    """Official VGGT histogram AUC: mean cumulative recall in 1-degree bins."""
    errors = np.asarray(errors)
    if not errors.size:
        return None
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.cumsum(histogram / errors.size).mean())


class GeometryMetrics:
    """Aggregate over all valid pixels/pairs, without averaging batch averages."""

    def __init__(self):
        self.samples = self.pixels = self.pairs = self.degenerate_pairs = 0
        self.point_pixels = 0
        self.sums = dict.fromkeys(
            [
                "abs_rel",
                "depth_sq",
                "delta1",
                "points",
                "points_m",
                "rotation",
                "translation",
            ],
            0.0,
        )
        self.rotation_errors = []
        self.translation_errors = []
        self.all_rotation_errors = []

    @torch.no_grad()
    def update(self, predictions, targets, scene_scale):
        values = [value for key, value in predictions.items() if key != "pose_enc_list"]
        values.extend(predictions.get("pose_enc_list", []))
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("Nonfinite model prediction during evaluation")
        valid = (
            targets["depth_valid"]
            & (targets["depth"] > 0)
            & torch.isfinite(targets["depth"])
        )
        pred_depth, gt_depth = predictions["depth"][valid], targets["depth"][valid]
        self.samples += targets["depth"].shape[0]
        self.pixels += valid.sum().item()
        self.sums["abs_rel"] += (
            ((pred_depth - gt_depth).abs() / gt_depth).double().sum().item()
        )
        self.sums["depth_sq"] += ((pred_depth - gt_depth).double() ** 2).sum().item()
        ratio = torch.maximum(
            pred_depth / gt_depth, gt_depth / pred_depth.clamp_min(1e-8)
        )
        self.sums["delta1"] += ((pred_depth > 0) & (ratio < 1.25)).sum().item()
        if "points_world" in predictions:
            self.point_pixels += valid.sum().item()
            point_error = (predictions["points_world"] - targets["points_world"]).norm(
                dim=2
            )
            pixel_valid = valid[:, :, 0]
            self.sums["points"] += point_error[pixel_valid].double().sum().item()
            self.sums["points_m"] += (
                (point_error * scene_scale[:, None, None, None])[pixel_valid]
                .double()
                .sum()
                .item()
            )
        rotation, translation, baseline_valid = relative_pose_errors(
            predictions["T_world_from_camera"], targets["T_world_from_camera"]
        )
        # Rotation metrics do not require a nonzero camera baseline.
        self.pairs += rotation.numel()
        self.sums["rotation"] += rotation.sum().item()
        self.all_rotation_errors.extend(rotation.flatten().cpu().tolist())
        self.degenerate_pairs += (~baseline_valid).sum().item()
        self.sums["translation"] += translation[baseline_valid].sum().item()
        self.rotation_errors.extend(rotation[baseline_valid].cpu().tolist())
        self.translation_errors.extend(translation[baseline_valid].cpu().tolist())

    def synchronize(self):
        """Merge pixel/pair sums and angle lists across disjoint evaluation shards."""
        if not dist.is_initialized():
            return
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, self.__dict__)
        for key in ("samples", "pixels", "pairs", "degenerate_pairs", "point_pixels"):
            setattr(self, key, sum(state[key] for state in states))
        self.sums = {
            key: sum(state["sums"][key] for state in states) for key in self.sums
        }
        for key in ("rotation_errors", "translation_errors", "all_rotation_errors"):
            setattr(self, key, [value for state in states for value in state[key]])

    def compute(self):
        if not self.samples or not self.pixels:
            raise ValueError("Evaluation requires samples with valid depth")
        count = len(self.translation_errors)
        errors = np.maximum(self.rotation_errors, self.translation_errors)
        return {
            "num_samples": self.samples,
            "num_valid_pixels": self.pixels,
            "num_pose_pairs": self.pairs,
            "num_degenerate_pose_pairs": self.degenerate_pairs,
            "depth_abs_rel": self.sums["abs_rel"] / self.pixels,
            "depth_rmse_normalized": (self.sums["depth_sq"] / self.pixels) ** 0.5,
            "depth_delta1": self.sums["delta1"] / self.pixels,
            **(
                {
                    "points_l2_normalized": self.sums["points"] / self.point_pixels,
                    "points_l2_m": self.sums["points_m"] / self.point_pixels,
                }
                if self.point_pixels
                else {}
            ),
            "rotation_error_deg": self.sums["rotation"] / self.pairs
            if self.pairs
            else None,
            "translation_error_deg": self.sums["translation"] / count
            if count
            else None,
            **{
                f"{name}_auc_{threshold}": pose_auc(angles, threshold)
                for name, angles in (
                    ("rotation", self.all_rotation_errors),
                    ("translation", self.translation_errors),
                    ("pose", errors),
                )
                for threshold in [3, 5, 15, 30]
            },
        }
