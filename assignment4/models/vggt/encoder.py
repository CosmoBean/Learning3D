"""Provided frozen DINOv2 feature encoder."""

from pathlib import Path

import torch
from filelock import FileLock
from torch import nn

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class FrozenDINO(nn.Module):
    """Convert RGB (N,3,H,W) in [0,1] into frozen DINOv2 patch tokens.

    Q3.3 fine-tuning: enable encoder gradients, include its parameters in the
    optimizer, and remove both forward's no_grad decorator and the no_grad
    block in MiniVGGT.forward. eval() itself does not disable gradients.
    """

    def __init__(
        self,
        repo=f"facebookresearch/dinov2:{DINO_REVISION}",
        weights=None,
        pretrained=True,
        name="dinov2_vits14",
    ):
        super().__init__()
        hub_dir = Path(torch.hub.get_dir())
        hub_dir.mkdir(parents=True, exist_ok=True)
        # Torch Hub downloads and extracts into shared paths without locking.
        with FileLock(str(hub_dir / "dinov2.lock")):
            self.backbone = torch.hub.load(
                repo,
                name,
                source="local" if Path(repo).is_dir() else "github",
                pretrained=pretrained and weights is None,
                trust_repo=True,
            )
        if weights is not None:
            self.backbone.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
        self.backbone.requires_grad_(False).eval()
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = 14
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb):
        return self.backbone.forward_features((rgb - self.mean) / self.std)[
            "x_norm_patchtokens"
        ]
