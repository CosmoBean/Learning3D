"""Shared helpers for Mini-LVSM training and evaluation."""

import importlib
import json
from pathlib import Path

import torch
from PIL import Image

from a4data.loading import make_dataloader


def load_config(path):
    """Read configs/lvsm.json and resolve the shared data config + overrides."""
    config = json.loads(Path(path).read_text())
    data = json.loads(Path(config["data_config"]).read_text())
    data.update(config.get("data_overrides", {}))
    config["data"] = data
    return config


def get_geometry():
    """Load the Part 1 geometry functions."""
    package = "a4data.geometry"
    return importlib.import_module(package)


def make_loader(config, split, **data_overrides):
    data = {**config["data"], **data_overrides}
    return make_dataloader(data, split=split, geometry=get_geometry())


def select_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config):
    package = "models.lvsm"
    model_class = importlib.import_module(package + ".model").MiniLVSM
    return model_class(**config.get("model", {}))


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


@torch.no_grad()
def save_visualization(path, inputs, targets, predictions, max_rows=4):
    """Save a PNG: one row per target view, columns = [sources..., GT, prediction].

    Uses the first sample of the batch only.
    """
    sources = inputs["source_rgb"][0]  # (S, 3, H, W)
    gt, pred = targets["rgb"][0], predictions["rgb"][0]  # (T, 3, H, W)
    rows = []
    for t in range(min(max_rows, gt.shape[0])):
        row = torch.cat([*sources, gt[t], pred[t]], dim=-1)  # (3, H, (S+2)*W)
        rows.append(row)
    grid = torch.cat(rows, dim=-2).clamp(0, 1)
    array = (grid.permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
