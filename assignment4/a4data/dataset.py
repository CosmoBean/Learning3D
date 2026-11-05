"""Model inputs and supervision for the shared online multi-view batch.

B: batch size; V: all views; S/T: LVSM source/target counts; H/W: image size.
Geometry maps are channel-first. RGB is in [0,1], geometry tensors are float32,
and depth_valid is bool. This module does not import geometry implementations.
"""

import torch


def _select_views(batch, key, indices):
    """Select (B, N, ...) from batch[key] (B, V, ...) using indices (B, N)."""
    value = batch[key]
    if value.ndim < 3 or indices.ndim != 2:
        raise ValueError("Task helpers require a collated batch from DataLoader")
    b = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[b, indices.to(value.device)]


def _lvsm_indices(batch, num_source_views):
    """Choose LVSM roles at the model boundary; leave shared geometry unchanged."""
    b, v = batch["rgb"].shape[:2]
    if not 1 <= num_source_views < v:
        raise ValueError("num_source_views must be >=1 and less than total views")
    # View order is already random; select source positions and use the rest as targets.
    sources = torch.linspace(0, v - 1, num_source_views).round().long()
    targets = torch.tensor(
        [i for i in range(v) if i not in sources.tolist()], dtype=torch.int64
    )
    return sources.expand(b, -1), targets.expand(b, -1)


def lvsm_inputs(batch, num_source_views):
    """Prepare LVSM inputs for pose-conditioned novel-view synthesis.

    Args:
        batch: Collated sample dict with rgb (B, V, 3, H, W),
            plucker (B, V, 6, H, W). RGB is in [0, 1]; rays use normalized world units.
        num_source_views: S sources at evenly spaced positions in the randomly
            sampled view order, not temporal order; T = V - S.

    Returns:
        source_rgb: (B, S, 3, H, W).
        source_plucker: (B, S, 6, H, W).
        target_plucker: (B, T, 6, H, W). Target RGB is supervision only.
    """
    sources, targets = _lvsm_indices(batch, num_source_views)
    return {
        "source_rgb": _select_views(batch, "rgb", sources),
        "source_plucker": _select_views(batch, "plucker", sources),
        "target_plucker": _select_views(batch, "plucker", targets),
    }


def lvsm_targets(batch, num_source_views):
    """Prepare LVSM RGB supervision.

    Args:
        batch: Collated dict with rgb (B, V, 3, H, W).
        num_source_views: Same source count used for lvsm_inputs.

    Returns:
        Dict with rgb (B, T, 3, H, W), target RGB in [0, 1].
    """
    _, indices = _lvsm_indices(batch, num_source_views)
    return {"rgb": _select_views(batch, "rgb", indices)}


def vggt_inputs(batch):
    """Prepare VGGT inputs from all images, without a source/target partition.

    Args:
        batch: Collated sample dict with rgb (B, V, 3, H, W), in [0, 1].

    Returns:
        Dict with rgb (B, V, 3, H, W). Annotated cameras/depth/points are not input.
    """
    return {"rgb": batch["rgb"]}


def vggt_targets(batch, include_points=False):
    """Prepare VGGT geometry supervision for all images, in input view order.

    Args:
        batch: Collated sample dict containing the fields and shapes below.
        include_points: Include point supervision when the optional point head is enabled.

    Returns:
        K: (B, V, 3, 3), pixel-space intrinsics.
        T_world_from_camera: (B, V, 4, 4), normalized camera-to-world poses.
        depth: (B, V, 1, H, W), normalized optical-axis z-depth.
        depth_valid: (B, V, 1, H, W), bool validity, not a foreground mask.
        points_world: (B, V, 3, H, W), optional normalized world-space point maps.
    """
    keys = ["K", "T_world_from_camera", "depth", "depth_valid"]
    if include_points:
        keys.append("points_world")
    return {key: batch[key] for key in keys}
