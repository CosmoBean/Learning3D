"""Provided evaluation loader: metric first-frame geometry, then sampled scale."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from a4data.paths import safe_path
from a4data.scale import compute_scene_scale

SCHEMA_VERSION = 1
FRAME_FIELDS = (
    "rgb",
    "K",
    "T_world_from_camera",
    "depth",
    "depth_valid",
    "points_world",
    "ray_origins",
    "ray_directions",
    "plucker",
)


class PreparedEvalDataset(Dataset):
    """Select V views from the first N saved pairs, with frame 0 always first.

    The package contains up to 128 chronologically ordered paired frames per
    sequence, rectified/cropped/resized and expressed in its first camera frame.
    Selection is deterministic per scene and seed, independent of workers.
    Only selected frame files are read. Scale is computed AFTER sampling using
    the same provided function as training. Student geometry is never called.
    """

    def __init__(
        self,
        root,
        split_file,
        split,
        num_views=8,
        num_frames=128,
        image_size=224,
        seed=825,
    ):
        if split not in ("val", "test"):
            raise ValueError("Prepared evaluation supports only val and test")
        if not 1 <= num_views <= num_frames:
            raise ValueError("Require eval_num_frames >= num_views >= 1")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported prepared evaluation schema")
        if manifest["coordinate_system"] != "first_frame_metric":
            raise ValueError(
                "Expected metric geometry in the first frame coordinate system"
            )
        if image_size != manifest["image_size"]:
            raise ValueError("image_size differs from the prepared evaluation package")
        splits = json.loads(Path(split_file).read_text())
        expected = sorted(name for name, part in splits.items() if part == split)
        self.scenes = []
        for name in expected:
            entry = manifest["scenes"].get(name)
            if entry is None or entry["split"] != split:
                raise ValueError(f"Prepared package is missing {split} scene {name}")
            if len(entry["frames"]) < num_frames:
                raise ValueError(
                    f"Scene {name} has fewer than {num_frames} prepared frames"
                )
            self.scenes.append((name, entry))
        if not self.scenes:
            raise ValueError(f"No prepared scenes for {split}")
        self.num_views, self.num_frames, self.seed = num_views, num_frames, seed

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        name, entry = self.scenes[index]
        # Use the scene ID to keep selection stable when dataset ordering changes.
        scene_seed = int.from_bytes(
            hashlib.sha256(name.encode()).digest()[:4], "little"
        )
        generator = torch.Generator().manual_seed(self.seed + scene_seed)
        selected = [0] + (
            torch.randperm(self.num_frames - 1, generator=generator)[
                : self.num_views - 1
            ]
            + 1
        ).tolist()
        frames = [entry["frames"][i] for i in selected]
        records = [
            torch.load(
                safe_path(self.root, f["path"]), map_location="cpu", weights_only=True
            )
            for f in frames
        ]
        sample = {
            key: torch.stack([record[key] for record in records])
            for key in FRAME_FIELDS
        }
        scale = compute_scene_scale(
            sample["points_world"].permute(0, 2, 3, 1), sample["depth_valid"]
        )
        sample["T_world_from_camera"][:, :3, 3] /= scale
        for key in ("depth", "points_world", "ray_origins"):
            sample[key] /= scale
        # Directions are unit length; moments scale with ray origins.
        sample["plucker"][:, 3:] /= scale
        sample["scene_scale"] = scale
        sample["original_world_from_reference"] = torch.tensor(
            entry["original_world_from_reference"], dtype=torch.float32
        )
        sample["frame_numbers"] = torch.tensor([f["frame_number"] for f in frames])
        sample["sequence_name"] = f"{name}_frame_{frames[0]['frame_number']:06d}"
        return sample
