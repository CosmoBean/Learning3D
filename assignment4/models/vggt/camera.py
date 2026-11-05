"""Provided camera conversions; no student TODOs."""

import torch


def decode_camera(camera, image_hw):
    """Decode (B,V,9) w2c translation, XYZW quaternion and vertical/horizontal FoV.

    Return c2w poses (B,V,4,4) and pixel K (B,V,3,3). Camera 0's identity
    pose is learned through supervision, rather than overwritten in decoding.
    """
    return decode_pose_encoding(camera.float(), image_hw)


def rotation_quaternion(rotation):
    """Return (...,4) XYZW quaternions using VGGT's nonnegative-W convention."""
    r = rotation
    a, b, c = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    squared = torch.stack(
        (1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1
    )
    roots = torch.zeros_like(squared)
    positive = squared > 0
    roots[positive] = squared[positive].sqrt()
    xy, xz, yz = (
        r[..., 0, 1] + r[..., 1, 0],
        r[..., 0, 2] + r[..., 2, 0],
        r[..., 1, 2] + r[..., 2, 1],
    )
    wx, wy, wz = (
        r[..., 2, 1] - r[..., 1, 2],
        r[..., 0, 2] - r[..., 2, 0],
        r[..., 1, 0] - r[..., 0, 1],
    )
    candidates = torch.stack(
        (
            torch.stack((squared[..., 0], wx, wy, wz), -1),
            torch.stack((wx, squared[..., 1], xy, xz), -1),
            torch.stack((wy, xy, squared[..., 2], yz), -1),
            torch.stack((wz, xz, yz, squared[..., 3]), -1),
        ),
        -2,
    ) / (2 * roots.clamp_min(0.1)[..., :, None])
    index = roots.argmax(-1)[..., None, None].expand(*roots.shape[:-1], 1, 4)
    q = candidates.gather(-2, index).squeeze(-2)[..., [1, 2, 3, 0]]
    return torch.where(q[..., 3:] < 0, -q, q)


def pose_encoding(poses, K, image_hw):
    """Encode (B,V,4,4) c2w and (B,V,3,3) K as (B,V,9) w2c T/quat/FoV."""
    rotation = poses[..., :3, :3].transpose(-1, -2)
    translation = -(rotation @ poses[..., :3, 3, None]).squeeze(-1)
    h, w = image_hw
    fov = torch.stack(
        (
            2 * torch.atan(h / (2 * K[..., 1, 1])),
            2 * torch.atan(w / (2 * K[..., 0, 0])),
        ),
        -1,
    )
    return torch.cat((translation, rotation_quaternion(rotation), fov), -1)


def decode_pose_encoding(encoded, image_hw):
    """Decode (B,V,9) w2c T/XYZW/FoV to c2w (B,V,4,4) and pixel K."""
    x, y, z, w = torch.nn.functional.normalize(encoded[..., 3:7], dim=-1).unbind(-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*encoded.shape[:-1], 3, 3)
    poses = (
        torch.eye(4, device=encoded.device, dtype=encoded.dtype)
        .expand(*encoded.shape[:-1], 4, 4)
        .clone()
    )
    poses[..., :3, :3] = rotation.transpose(-1, -2)
    poses[..., :3, 3] = -(rotation.transpose(-1, -2) @ encoded[..., :3, None]).squeeze(
        -1
    )
    h, w = image_hw
    K = (
        torch.eye(3, device=encoded.device, dtype=encoded.dtype)
        .expand(*encoded.shape[:-1], 3, 3)
        .clone()
    )
    # Keep decoded intrinsics finite even if an early FoV prediction reaches zero.
    # Camera supervision uses the original encodings, not these clamped FoVs.
    fov = encoded[..., 7:].clamp(1e-4, torch.pi - 1e-4)
    K[..., 0, 0] = (w / 2) / torch.tan(fov[..., 1] / 2)
    K[..., 1, 1] = (h / 2) / torch.tan(fov[..., 0] / 2)
    K[..., 0, 2], K[..., 1, 2] = w / 2, h / 2
    return poses, K
