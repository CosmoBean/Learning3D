"""Online ScanNet++ sampling and geometry preparation, without data caches."""

import importlib
import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset, get_worker_info

from a4data import geometry as default_geometry
from a4data.paths import safe_path
from a4data.preprocessing import prepare_sample
from a4data.scannetpp.io import load_raw_sequence


class ScanNetPPDataset(Dataset):
    """Read paired frames and compute one multi-view sample in __getitem__.

    Args:
        root: Extracted <scene>/iphone/{images,depths,cameras.json} data.
        split_file: JSON mapping every scene ID to train, val or test.
        split: Scene split to load. Train draws new random views on each call;
            val/test repeat seeded selections for each scene.
        num_views: V selected views, independent of model source/target roles.
        sampling_range: Maximum consecutive available frames in the sampling
            window. Select num_views distinct frames in random order within it.
        image_size: Output H=W in pixels.
        seed: Train sampling seed; DataLoader workers have separate RNG streams.
        geometry: Module implementing the Q1 geometry functions.

    Returns the channel-first tensors described in docs/DATA_CONTRACT.md.
    Only camera metadata and frame IDs are kept in memory. RGB/depth and geometry
    are loaded/computed on demand and are not written to disk.
    """

    def __init__(
        self,
        root,
        split_file,
        split="train",
        num_views=8,
        sampling_range=32,
        image_size=224,
        seed=825,
        geometry=default_geometry,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if num_views < 1 or sampling_range < num_views or image_size < 2:
            raise ValueError(
                "Require sampling_range >= num_views >=1 and image_size >=2"
            )
        self.root = Path(root)
        self.split = split
        self.num_views = num_views
        self.sampling_range = sampling_range
        self.image_size = image_size
        self.seed = seed
        # Module names remain picklable with spawn-based DataLoader workers.
        self.geometry_module = geometry.__name__
        self._rng = None
        self._worker_seed = None
        self.scenes = []
        splits = json.loads(Path(split_file).read_text())
        for name, scene_split in sorted(splits.items()):
            if scene_split != split:
                continue
            folder = safe_path(self.root, f"{name}/iphone")
            cameras = json.loads((folder / "cameras.json").read_text())
            # Keep all paired frames; sampling happens only in __getitem__.
            images = {p.name for p in (folder / "images").glob("*.jpg")}
            depths = {p.stem for p in (folder / "depths").glob("*.png")}
            frames = sorted(
                int(Path(e["image_name"]).stem.split("_")[-1])
                for e in cameras["extrinsics"]
                if e["image_name"] in images and Path(e["image_name"]).stem in depths
            )
            if len(frames) < num_views:
                warnings.warn(
                    f"Skipping {name}: need {num_views} paired frames, found {len(frames)}",
                    stacklevel=2,
                )
                continue
            self.scenes.append((name, frames, cameras))
        if not self.scenes:
            raise ValueError(f"No scenes with enough paired frames for split {split}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index, stage="full", fixed_sampling=False):
        """Fixed sampling is used only for reproducible geometry visualizations."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        name, frames, cameras = self.scenes[index]
        window_size = min(self.sampling_range, len(frames))
        if self.split == "train" and not fixed_sampling:
            worker = get_worker_info()
            worker_seed = self.seed + (worker.seed if worker else 0)
            if self._worker_seed != worker_seed:
                self._rng = torch.Generator().manual_seed(worker_seed)
                self._worker_seed = worker_seed
            rng = self._rng
        else:
            rng = torch.Generator().manual_seed(self.seed + index)
        start = torch.randint(len(frames) - window_size + 1, (), generator=rng).item()
        offsets = torch.randperm(window_size, generator=rng)[: self.num_views].tolist()
        selected = [frames[start + offset] for offset in offsets]
        raw = load_raw_sequence(self.root, name, selected, cameras=cameras)
        sample = prepare_sample(
            raw,
            selected,
            self.image_size,
            importlib.import_module(self.geometry_module),
            stage=stage,
        )
        sample["sequence_name"] = f"{name}_frame_{selected[0]:06d}"
        return sample
