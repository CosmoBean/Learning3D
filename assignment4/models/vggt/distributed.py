"""Small torchrun helpers; batch size and worker count are global settings."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset


def setup(device):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"
    return 0, 1, device


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def shard_loader(loader, config, split):
    """Disjoint eval shards without padding; standard shuffled DDP train sampler."""
    if not dist.is_initialized():
        return loader
    rank, world = dist.get_rank(), dist.get_world_size()
    if config["batch_size"] % world:
        raise ValueError("Global batch_size must be divisible by GPU count")
    dataset = loader.dataset
    sampler = None
    if split == "train":
        sampler = DistributedSampler(dataset, shuffle=True, seed=config["seed"])
    else:
        dataset = Subset(dataset, list(range(rank, len(dataset), world)))
    workers = max(1, config["num_workers"] // world) if config["num_workers"] else 0
    return DataLoader(
        dataset,
        batch_size=config["batch_size"] // world,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )


def sum_values(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.tolist()
