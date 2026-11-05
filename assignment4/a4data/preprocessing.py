"""Shared resizing and geometry preparation for multi-view samples."""

import torch
import torch.nn.functional as F


def resize_sample(raw, image_size, geometry):
    """Center-crop and resize RGB/depth together, updating intrinsics."""
    keys = ["rgb", "depth", "depth_valid", "K"]
    sample = {key: [] for key in keys}
    for view, image in enumerate(raw["rgb"]):
        h, w = image.shape[-2:]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        image = image[:, top : top + side, left : left + side]
        K_cropped = raw["K"][view].clone()
        K_cropped[0, 2] -= left
        K_cropped[1, 2] -= top
        sample["K"].append(
            geometry.resize_intrinsics(
                K_cropped, (side, side), (image_size, image_size)
            )
        )
        sample["rgb"].append(
            F.interpolate(
                image[None],
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0].clamp(0, 1)
        )
        for key in ["depth", "depth_valid"]:
            cropped = raw[key][view][:, top : top + side, left : left + side]
            value = F.interpolate(
                cropped[None].float(),
                size=(image_size, image_size),
                mode="nearest-exact",
            )[0]
            sample[key].append(value.bool() if key != "depth" else value)
    sample = {key: torch.stack(value) for key, value in sample.items()}
    sample["depth_valid"] &= torch.isfinite(sample["depth"]) & (sample["depth"] > 0)
    sample["depth"] = torch.where(sample["depth_valid"], sample["depth"], 0)
    return sample


def prepare_sample(raw, frame_numbers, image_size, geometry, stage="full"):
    """Prepare a sample, optionally stopping at a visualization stage.

    Args:
        raw: Dict of V-element tensor lists. Each view has rgb (3, H_in, W_in),
            depth/depth_valid (1, H_in, W_in), K (3, 3), and
            T_world_from_camera (4, 4). Depth and translations are in meters.
        frame_numbers: V selected frame IDs, in sampled view order. Camera 0
            and all valid points define the shared reference frame and scale.
        image_size: Integer output size; H = W = image_size.
        geometry: Module providing the five Q1 geometry functions.
        stage: full for model inputs; unprojection or canonicalization for debugging.

    Returns:
        Channel-first sample dict documented in docs/DATA_CONTRACT.md.
        RGB/depth/point shapes are (V, 3, H, W), (V, 1, H, W), (V, 3, H, W).
        The geometry functions return channel-last maps (V, H, W, 3/6), which
        are rearranged here. VGGT uses all V views, including their supervision.
    """
    if stage not in ("full", "unprojection", "canonicalization"):
        raise ValueError(f"Unknown geometry stage: {stage}")
    sample = resize_sample(raw, image_size, geometry)
    sample["frame_numbers"] = torch.tensor(frame_numbers, dtype=torch.int64)
    if stage == "unprojection":
        poses = torch.stack(raw["T_world_from_camera"])
        points = geometry.unproject_depth(sample["K"], poses, sample["depth"])
        points = torch.where(sample["depth_valid"].permute(0, 2, 3, 1), points, 0)
        sample["T_world_from_camera"] = poses
        sample["points_world"] = points.permute(0, 3, 1, 2).contiguous()
        return sample
    poses, depth, points, reference, scale = geometry.canonicalize_cameras(
        sample["K"],
        torch.stack(raw["T_world_from_camera"]),
        sample["depth"],
        sample["depth_valid"],
    )
    sample.update(
        T_world_from_camera=poses,
        depth=depth,
        original_world_from_reference=reference,
        scene_scale=scale,
    )
    sample["points_world"] = points.permute(0, 3, 1, 2).contiguous()
    if stage == "canonicalization":
        return sample
    origins, directions = geometry.camera_rays(
        sample["K"], poses, (image_size, image_size)
    )
    sample["ray_origins"] = origins.permute(0, 3, 1, 2).contiguous()
    sample["ray_directions"] = directions.permute(0, 3, 1, 2).contiguous()
    sample["plucker"] = (
        geometry.plucker_coordinates(origins, directions)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return sample
