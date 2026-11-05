"""Complete the TODOs for Part 1 of the assignment.

Fill the None placeholders and remove each NotImplementedError.

V is the number of views; H/W are image height/width.
Tensors are float32 unless stated otherwise. Cameras use OpenCV
axes (+X right, +Y down, +Z forward) and column-vector transforms. These functions
operate on one multi-view sample, without a batch dimension. Pixel centers are
integer (u, v), with (0, 0) at the top-left; u is the column and v is the row.
Geometry functions return channel-last maps; the provided pipeline converts
them to channel-first maps for the dataloader.
"""

# TODO intermediate variables are intentionally unused until implemented.
# ruff: noqa: F841, RUF059

import torch

from a4data.scale import compute_scene_scale


def _pixel_grid(H: int, W: int, like: torch.Tensor) -> torch.Tensor:
    """Homogeneous pixel centers (u, v, 1) with shape (H, W, 3)."""
    v, u = torch.meshgrid(
        torch.arange(H, dtype=like.dtype, device=like.device),
        torch.arange(W, dtype=like.dtype, device=like.device),
        indexing="ij",
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=-1)


def resize_intrinsics(
    K: torch.Tensor, input_hw: tuple[int, int], output_hw: tuple[int, int]
) -> torch.Tensor:
    """Q1.1: Update pixel-space intrinsics for a full-image resize.

    Args:
        K: (3, 3), input intrinsics in pixels; crop adjustment is already provided.
        input_hw: (H_in, W_in), image size after the provided center crop.
        output_hw: (H_out, W_out), resized image size in pixels.

    Returns:
        K_resized: (3, 3), intrinsics for the resized image. Match the provided
            align_corners=False resize: for each image axis,
            x_out + 0.5 = (size_out / size_in) * (x_in + 0.5).
    """
    # TODO (Q1.1): Update focal lengths and principal point for the resize.
    sy = output_hw[0] / input_hw[0]
    sx = output_hw[1] / input_hw[1]
    K_resized = K.clone()  # (3, 3)
    K_resized[0, 0] = K[0, 0] * sx
    K_resized[1, 1] = K[1, 1] * sy
    K_resized[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    K_resized[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K_resized


def unproject_depth(
    K: torch.Tensor, T_world_from_camera: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    """Q1.2: Unproject optical-axis z-depth into world-space points.

    Args:
        K: (V, 3, 3), resized pixel-space intrinsic matrices.
        T_world_from_camera: (V, 4, 4), camera-to-world poses. Translations
            must use the same units as depth (m before canonicalization).
        depth: (V, 1, H, W), optical-axis z-depth, not distance along
            a unit ray. Invalid pixels are zero; the provided pipeline zeros
            their output points after this function returns.

    Returns:
        points_world: (V, H, W, 3), world points in the same units as depth.
    """
    H, W = depth.shape[-2:]

    # TODO (Q1.2): Build homogeneous pixel coordinates.
    pixels = _pixel_grid(H, W, depth)  # (H, W, 3)

    # TODO (Q1.2): Unproject pixels to camera-space points using K and z-depth.
    rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(K), pixels)
    points_camera = rays * depth[:, 0, :, :, None]  # (V, H, W, 3)

    # TODO (Q1.2): Transform camera-space points into world coordinates.
    R, t = T_world_from_camera[:, :3, :3], T_world_from_camera[:, :3, 3]
    points_world = torch.einsum("vij,vhwj->vhwi", R, points_camera) + t[:, None, None]
    return points_world


def canonicalize_cameras(
    K: torch.Tensor,
    T_world_from_camera: torch.Tensor,
    depth: torch.Tensor,
    depth_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q1.3: Canonicalize geometry using the first camera and valid-point centroid.

    Args:
        K: (V, 3, 3), resized pixel-space intrinsics.
        T_world_from_camera: (V, 4, 4), camera-to-world poses; translations in m.
        depth: (V, 1, H, W), optical-axis z-depth in m; invalid pixels are zero.
        depth_valid: (V, 1, H, W), bool validity across all views.

    Returns:
        poses_normalized: (V, 4, 4), camera-to-first-frame poses with scaled
            translations. Camera 0 is identity.
        depth_normalized: (V, 1, H, W), scaled z-depth; invalid pixels are zero.
        points_normalized: (V, H, W, 3), scaled first-frame points; invalid
            pixels are zero. The valid-point centroid has distance 1 from origin.
        original_world_from_reference: (4, 4), original pose of camera 0.
        scene_scale: (), centroid distance in m before scaling: ||mean(p_valid)||.

    Camera 0 is the first sampled view. Transform cameras to its coordinate
    frame, then unproject depth using your Q1.2 function. Call the provided
    compute_scene_scale to obtain the valid-point centroid's distance from
    the origin. Divide translations, depth and points by this scale;
    do not subtract the centroid.
    """
    # TODO (Q1.3): Choose camera 0 as the reference and transform all poses.
    original_world_from_reference = T_world_from_camera[0]  # (4, 4)
    # Compute relative transforms in float64 for precision, then cast to float32.
    poses_relative = (
        torch.linalg.inv(original_world_from_reference.double())
        @ T_world_from_camera.double()
    ).to(T_world_from_camera.dtype)  # (V, 4, 4)

    # Use your Q1.2 implementation to compute points in the first camera's frame.
    points_reference = unproject_depth(K, poses_relative, depth)  # (V, H, W, 3)

    # Provided: use exactly the same scale definition as the evaluation loader.
    scene_scale = compute_scene_scale(points_reference, depth_valid)

    # TODO (Q1.3): Divide translations, depth and points by the same scale.
    # Keep camera 0 at the origin; do not subtract the point-cloud centroid.
    poses_normalized = poses_relative.clone()  # (V, 4, 4)
    poses_normalized[:, :3, 3] /= scene_scale
    depth_normalized = torch.where(depth_valid, depth / scene_scale, 0)  # (V, 1, H, W)
    valid = depth_valid[:, 0, :, :, None]
    points_normalized = torch.where(valid, points_reference / scene_scale, 0)  # (V, H, W, 3)

    return (
        poses_normalized,
        depth_normalized,
        points_normalized,
        original_world_from_reference,
        scene_scale,
    )


def camera_rays(
    K: torch.Tensor,
    T_world_from_camera: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q1.4: Construct one world-space ray for every pixel of every view.

    Args:
        K: (V, 3, 3), resized pixel-space intrinsic matrices.
        T_world_from_camera: (V, 4, 4), canonicalized camera-to-world poses.
        image_hw: (H, W), output image size in pixels. Pixel centers are integer
            (u, v), with the top-left pixel at (0, 0).

    Returns:
        origins: (V, H, W, 3), camera centers broadcast over all pixels, in
            canonical world units.
        directions: (V, H, W, 3), unit ray directions in world coordinates.
    """
    H, W = image_hw

    # TODO (Q1.4): Build homogeneous pixel coordinates for the image grid.
    pixels = _pixel_grid(H, W, K)  # (H, W, 3)

    # TODO (Q1.4): Convert pixels to camera-space ray directions using K.
    camera_directions = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(K), pixels)

    # TODO (Q1.4): Rotate directions to world space and normalize each vector.
    R = T_world_from_camera[:, :3, :3]
    directions = torch.einsum("vij,vhwj->vhwi", R, camera_directions)
    directions = torch.nn.functional.normalize(directions, dim=-1)  # (V, H, W, 3)

    # TODO (Q1.4): Broadcast each camera center as the ray origins.
    origins = T_world_from_camera[:, None, None, :3, 3].expand_as(directions)
    return origins, directions


def plucker_coordinates(
    origins: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Q1.5: Compute the six-channel Plucker representation of each ray.

    Args:
        origins: (V, H, W, 3), world-space ray origins in canonical units.
        directions: (V, H, W, 3), unit world-space ray directions.

    Returns:
        plucker: (V, H, W, 6), direction followed by moment: [d, m].
            Use the convention m = o x d.
    """
    # TODO (Q1.5): Compute the ray moments.
    moments = torch.linalg.cross(origins, directions, dim=-1)  # (V, H, W, 3)

    # TODO (Q1.5): Concatenate directions and moments along the last dimension.
    plucker = torch.cat([directions, moments], dim=-1)  # (V, H, W, 6)
    return plucker
