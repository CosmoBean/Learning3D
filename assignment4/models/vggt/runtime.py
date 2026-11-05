"""Small shared setup for Mini-VGGT training and evaluation."""

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from a4data.dataset import vggt_inputs
from a4data.loading import DATA_OPTIONS, make_dataloader
from a4data.preprocessing import prepare_sample
from a4data.scannetpp.io import load_raw_sequence


def add_arguments(parser):
    parser.add_argument("--config", help="Model config; defaults to configs/vggt.json")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--predict-points", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--encoder-repo", help="Optional local DINOv2 checkout")
    parser.add_argument("--encoder-weights", help="Optional local DINOv2 checkpoint")
    for key, kind in DATA_OPTIONS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=kind)


def load_config(args, checkpoint_config=None):
    if args.config or checkpoint_config is None:
        config = json.loads(Path(args.config or "configs/vggt.json").read_text())
        data = json.loads(Path(config["data_config"]).read_text())
        data.update(config["data_overrides"])
    else:
        config = json.loads(json.dumps(checkpoint_config))
        data = config["data"].copy()
    for key in DATA_OPTIONS:
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    config["data"] = data
    config["model"]["image_size"] = data["image_size"]
    for key in ["encoder_repo", "encoder_weights"]:
        if getattr(args, key) is not None:
            config["model"][key] = getattr(args, key)
    if args.predict_points is not None:
        config["model"]["predict_points"] = args.predict_points
    return config


def build_components(config, from_checkpoint=False):
    package = "models.vggt"
    model_options = config["model"].copy()
    if from_checkpoint:
        model_options.update(encoder_pretrained=False, encoder_weights=None)
    model = importlib.import_module(package + ".model").MiniVGGT(**model_options)
    loss_fn = importlib.import_module(package + ".losses").compute_loss
    geometry = importlib.import_module(
        "a4data.geometry"
    )
    return model, loss_fn, geometry


def fixed_training_loader(data, geometry, count=2, selection=None):
    """Freeze train scenes AND sampled frame order in RAM for overfitting.

    Saved scene/frame selections can be replayed for checkpoint evaluation.
    This does not create a data cache and never substitutes validation scenes.
    """
    dataset = make_dataloader(data, "train", geometry).dataset
    samples = []
    if selection is None:
        if count < 1 or count > len(dataset):
            raise ValueError("overfit-scenes must be within the training scene count")
        selection = []
        for i in range(count):
            sample = dataset[i]
            samples.append(sample)
            selection.append(
                {
                    "scene_name": dataset.scenes[i][0],
                    "frame_numbers": sample["frame_numbers"].tolist(),
                }
            )
    else:
        cameras = {name: camera for name, _, camera in dataset.scenes}
        for entry in selection:
            name, frames = entry["scene_name"], entry["frame_numbers"]
            raw = load_raw_sequence(data["root"], name, frames, cameras[name])
            sample = prepare_sample(raw, frames, data["image_size"], geometry)
            sample["sequence_name"] = f"{name}_frame_{frames[0]:06d}"
            samples.append(sample)
    loader = DataLoader(
        samples,
        batch_size=data["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(data["seed"]),
    )
    return loader, selection


def to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict(model, batch, device):
    """Move only RGB to the model; use BF16 attention on CUDA, FP32 on CPU."""
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        return model(to_device(vggt_inputs(batch), device))
