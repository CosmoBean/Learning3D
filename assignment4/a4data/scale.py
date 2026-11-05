"""Provided scene-scale calculation, shared by training and evaluation."""

import torch


def compute_scene_scale(points_reference: torch.Tensor, depth_valid: torch.Tensor):
    """Distance from the reference camera to the valid-point centroid.

    points_reference: (V,H,W,3), in meters; depth_valid: (V,1,H,W), bool.
    Pool valid pixels across all selected views and accumulate in float64.
    Return a scalar with the points' dtype and device.
    """
    valid = depth_valid[:, 0] & torch.isfinite(points_reference).all(dim=-1)
    if not valid.any():
        raise ValueError("Cannot normalize a scene with no valid points")
    scale = points_reference[valid].double().mean(0).norm().to(points_reference.dtype)
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Valid-point centroid has a degenerate distance from origin")
    return scale
