"""Geometric consistency checks for multi-view samples."""

import torch


def validate_sample(sample, atol=2e-4):
    rgb = sample["rgb"]
    v, c, h, w = rgb.shape
    if c != 3 or h != w:
        raise ValueError("Expected square RGB views [V,3,H,W]")
    shapes = {
        "depth_valid": (v, 1, h, w),
        "depth": (v, 1, h, w),
        "K": (v, 3, 3),
        "T_world_from_camera": (v, 4, 4),
        "ray_origins": (v, 3, h, w),
        "ray_directions": (v, 3, h, w),
        "plucker": (v, 6, h, w),
        "points_world": (v, 3, h, w),
    }
    for key, shape in shapes.items():
        value = sample[key]
        if tuple(value.shape) != shape:
            raise ValueError(f"Wrong {key} shape: {value.shape}; expected {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}")
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must be finite and in [0,1]")
    if sample["depth_valid"].dtype != torch.bool:
        raise ValueError("depth_valid must be boolean")
    if not sample["depth_valid"].any() or not torch.all(
        sample["depth"][sample["depth_valid"]] > 0
    ):
        raise ValueError("Missing positive valid depth")
    poses = sample["T_world_from_camera"]
    rotations = poses[:, :3, :3]
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations,
        torch.eye(3).expand(v, 3, 3),
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(v), atol=atol, rtol=0
    )
    torch.testing.assert_close(
        poses[:, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(v, 4), atol=atol, rtol=0
    )
    torch.testing.assert_close(poses[0], torch.eye(4), atol=atol, rtol=0)
    torch.testing.assert_close(
        sample["K"][:, 2], torch.tensor([0.0, 0.0, 1.0]).expand(v, 3), atol=atol, rtol=0
    )
    if not torch.all(sample["K"][:, [0, 1], [0, 1]] > 0):
        raise ValueError("Focal lengths must be positive")
    directions = sample["ray_directions"]
    torch.testing.assert_close(
        directions.norm(dim=1), torch.ones(v, h, w), atol=atol, rtol=0
    )
    torch.testing.assert_close(
        sample["ray_origins"],
        poses[:, :3, 3, None, None].expand(v, 3, h, w),
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(sample["plucker"][:, :3], directions, atol=atol, rtol=0)
    torch.testing.assert_close(
        (sample["plucker"][:, 3:] * directions).sum(1),
        torch.zeros(v, h, w),
        atol=atol,
        rtol=0,
    )
    points = sample["points_world"].permute(0, 2, 3, 1)
    camera_points = torch.einsum(
        "vji,vhwj->vhwi", rotations, points - poses[:, None, None, :3, 3]
    )
    projected = torch.einsum("vij,vhwj->vhwi", sample["K"], camera_points)
    valid = sample["depth_valid"][:, 0]
    centroid_distance = points[valid].double().mean(dim=0).norm().float()
    torch.testing.assert_close(centroid_distance, torch.tensor(1.0), atol=atol, rtol=0)
    pixels = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    expected = torch.stack((x, y), -1).float().expand(v, h, w, 2)
    camera_directions = torch.einsum(
        "vji,vhwj->vhwi", rotations, directions.permute(0, 2, 3, 1)
    )
    if not torch.all(camera_directions[..., 2] > 0):
        raise ValueError("Camera-space rays must point forward")
    projected_rays = torch.einsum("vij,vhwj->vhwi", sample["K"], camera_directions)
    ray_pixels = projected_rays[..., :2] / projected_rays[..., 2:]
    ray_reprojection_error = (ray_pixels - expected).norm(dim=-1)
    if ray_reprojection_error.max() > 0.01:
        raise ValueError(
            f"Ray reprojection exceeds 0.01 pixels: {ray_reprojection_error.max().item()}"
        )
    reprojection_error = (pixels[valid] - expected[valid]).norm(dim=-1)
    torch.testing.assert_close(
        camera_points[..., 2][valid], sample["depth"][:, 0][valid], atol=atol, rtol=atol
    )
    if reprojection_error.max() > 0.01:
        raise ValueError(
            f"Point reprojection exceeds 0.01 pixels: {reprojection_error.max().item()}"
        )
    return {
        "views": v,
        "resolution": h,
        "valid_depth_fraction": valid.float().mean().item(),
        "max_reprojection_error_px": reprojection_error.max().item(),
        "max_ray_reprojection_error_px": ray_reprojection_error.max().item(),
        "centroid_distance": centroid_distance.item(),
    }
