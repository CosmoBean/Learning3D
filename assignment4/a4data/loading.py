"""Shared training and prepared evaluation loaders for models and checks."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from a4data import geometry as default_geometry
from a4data.scannetpp.dataset import ScanNetPPDataset
from a4data.scannetpp.prepared import PreparedEvalDataset

DEFAULT_CONFIG = "configs/scannetpp/iphone/data.json"
DATA_OPTIONS = {
    "root": str,
    "split_file": str,
    "num_views": int,
    "sampling_range": int,
    "image_size": int,
    "batch_size": int,
    "num_workers": int,
    "seed": int,
    "eval_root": str,
    "eval_num_frames": int,
}


def add_data_arguments(parser):
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    for key, kind in DATA_OPTIONS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=kind)


def data_config(args):
    """Read a plain JSON config and apply explicit CLI overrides."""
    config = json.loads(Path(args.config).read_text())
    for key in DATA_OPTIONS:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def make_dataloader(config, split="train", geometry=default_geometry):
    """Train online; use the provided prepared loader for configured val/test.

    Without eval_root, validation/test also use online preprocessing. A configured
    missing evaluation package raises an error instead of falling back to Q1 code.
    """
    if split in ("val", "test") and config.get("eval_root"):
        dataset = PreparedEvalDataset(
            root=config["eval_root"],
            split_file=config["split_file"],
            split=split,
            num_views=config["num_views"],
            image_size=config["image_size"],
            num_frames=config.get("eval_num_frames", 128),
            seed=config["seed"],
        )
    else:
        dataset = ScanNetPPDataset(
            **{
                key: config[key]
                for key in (
                    "root",
                    "split_file",
                    "num_views",
                    "sampling_range",
                    "image_size",
                    "seed",
                )
            },
            split=split,
            geometry=geometry,
        )
    workers = config["num_workers"]
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=split == "train",
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
