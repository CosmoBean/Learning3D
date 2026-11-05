"""Read the provided ScanNet++ iPhone images, millimeter depth and cameras.

cameras.json stores PyTorch3D NDC intrinsics and row-vector world-to-camera
extrinsics, converted from the registered COLMAP cameras. Prepared RGB is
resized but distorted; RGB and aligned depth are rectified together.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from a4data.paths import safe_path


def opencv_camera(intrinsics, extrinsics, image_hw):
    """Convert a prepared camera record to OpenCV column-vector conventions.

    Args:
        intrinsics: Dict with focal_length (2,), principal_point (2,) in
            PyTorch3D NDC and image_size (2,) in [width, height] pixel order.
        extrinsics: Dict with R (3, 3) and T (3,), PyTorch3D row-vector
            world-to-camera rotation and translation (meters).
        image_hw: (H, W), stored RGB/depth resolution in pixels.

    Returns:
        K: (3, 3), float32 pixel-space intrinsic matrix at stored resolution.
        pose: (4, 4), float32 OpenCV camera-to-world pose, translation in meters.
    """
    width, height = intrinsics["image_size"]
    focal = np.asarray(intrinsics["focal_length"], dtype=np.float32)
    principal = np.asarray(intrinsics["principal_point"], dtype=np.float32)
    scale = min(width, height) / 2
    K = np.array(
        [
            [focal[0] * scale, 0, width / 2 - principal[0] * scale],
            [0, focal[1] * scale, height / 2 - principal[1] * scale],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # The supplied images have been resized from the annotated resolution.
    h, w = image_hw
    sx, sy = w / width, h / height
    K = (
        np.array(
            [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float32
        )
        @ K
    )
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    R = flip @ np.asarray(extrinsics["R"], dtype=np.float32).T
    t = flip @ np.asarray(extrinsics["T"], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = -R.T @ t
    return K, pose


def load_raw_sequence(root, scene_name, frame_numbers, cameras=None):
    """Read and rectify selected paired frames before the student geometry tasks.

    Args:
        root: Prepared data root containing <scene>/iphone/ directories.
        scene_name: Scene ID string.
        frame_numbers: (V,), list of original integer frame IDs, in output order.
        cameras: Optional preloaded cameras.json dictionary.

    Returns:
        Dict of V-element lists, with one tensor per view:
            rgb: (3, H, W), float32 RGB in [0, 1].
            depth: (1, H, W), float32 optical-axis z-depth in meters.
            depth_valid: (1, H, W), bool finite-positive depth validity.
            K: (3, 3), float32 rectified pixel-space intrinsic matrix.
            T_world_from_camera: (4, 4), float32 OpenCV camera-to-world pose.
        H/W are the stored frame dimensions, before resizing to 224x224.
    """
    folder = safe_path(root, f"{scene_name}/iphone")
    if cameras is None:
        cameras = json.loads((folder / "cameras.json").read_text())
    frames = {
        int(Path(e["image_name"]).stem.split("_")[-1]): e for e in cameras["extrinsics"]
    }
    result = {
        key: [] for key in ["rgb", "depth", "depth_valid", "K", "T_world_from_camera"]
    }
    for number in frame_numbers:
        frame = frames[number]
        image_path = safe_path(folder / "images", frame["image_name"])
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(folder / "depths" / (image_path.stem + ".png")) as image:
            depth = np.asarray(image, dtype=np.float32) / 1000
        h, w = rgb.shape[:2]
        if depth.shape != (h, w):
            raise ValueError("RGB and depth must use the same stored resolution")
        intrinsic = cameras["intrinsics"][str(frame["camera_id"])]
        if intrinsic["camera_type"] != "OPENCV":
            raise ValueError("Expected an OPENCV camera")
        K, pose = opencv_camera(intrinsic, frame, (h, w))
        distortion = np.asarray(intrinsic["distortion"], dtype=np.float32).reshape(-1)
        if np.any(distortion):
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 0, (w, h))
            map_x, map_y = cv2.initUndistortRectifyMap(
                K, distortion, None, new_K, (w, h), cv2.CV_32FC1
            )
            rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)
            K = new_K.astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0).astype(np.float32)
        result["rgb"].append(
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        )
        result["depth"].append(torch.from_numpy(depth[None]))
        result["depth_valid"].append(torch.from_numpy(valid[None]))
        result["K"].append(torch.from_numpy(K))
        result["T_world_from_camera"].append(torch.from_numpy(pose))
    return result
