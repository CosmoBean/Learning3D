"""Q3.0: Camera and dense depth/confidence prediction heads."""


import math

import torch
import torch.nn.functional as F
from torch import nn

from models.vggt.attention import GeometryBlock


def patches_to_image(values, image_hw, patch_size, channels):
    """Provided: (B,V,P,channels*p*p) -> (B,V,channels,H,W).

    P=(H/p)*(W/p); patches are ordered row-first on the image grid.
    """
    b, v = values.shape[:2]
    h, w = image_hw
    p = patch_size
    return (
        values.reshape(b, v, h // p, w // p, channels, p, p)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .reshape(b, v, channels, h, w)
    )


class CameraHead(nn.Module):
    def __init__(self, width, num_heads=4, num_layers=4):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.trunk = nn.ModuleList(
            [GeometryBlock(width, num_heads) for _ in range(num_layers)]
        )
        self.trunk_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 9)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[6] = 1.0
            self.mlp[-1].bias[7:] = 2 * math.atan(0.5)

    def forward(self, camera_features):
        """Provided: cross-view transformer interaction, then camera prediction.

        camera_features: (B,V,2D); view 0 is already marked by its camera token.
        Zero positions disable spatial RoPE for camera-token attention.
        """
        tokens = self.token_norm(camera_features)
        positions = tokens.new_zeros(tokens.shape[1], 2)
        for block in self.trunk:
            tokens = block(tokens, positions)
        return self.predict(self.trunk_norm(tokens))

    def predict(self, camera_features):
        """Q3.0: (B,V,2D) refined camera features -> (B,V,9) pose encoding.

        Apply self.mlp to predict w2c translation (3), XYZW quaternion (4),
        and vertical/horizontal FoV (2). Leave translation/quaternion linear;
        apply ReLU only to the FoV channels. Compute the activation in float32.
        Do not normalize or flip the quaternion before the camera loss.
        """
        # TODO (Q3.0): Apply self.mlp and the FoV activation above.
        out = self.mlp(camera_features).float()
        return torch.cat([out[..., :7], F.relu(out[..., 7:])], dim=-1)


class DepthHead(nn.Module):
    def __init__(self, width, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.channels = 1
        self.dense = nn.Linear(width, self.channels * patch_size**2)
        self.confidence = nn.Linear(width, patch_size**2)
        self.refine = nn.Sequential(
            nn.Conv2d(self.channels + 3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, self.channels, 3, padding=1),
        )
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, patches, rgb):
        """Q3.0: Patch features and RGB -> dense depth/confidence dictionary.

        Args:
            patches: (B,V,P,2D), concatenated frame/global patch features only.
            rgb: (B,V,3,H,W), input images in [0,1]; H/W divisible by p.

        Returns:
            depth: (B,V,1,H,W), exp(raw_depth).
            depth_conf: (B,V,1,H,W), 1 + exp(raw_confidence).
        """
        b, v, _, h, w = rgb.shape
        # TODO (Q3.0): Apply self.dense and the provided patches_to_image.
        p = self.patch_size
        dense = patches_to_image(self.dense(patches), (h, w), p, self.channels)
        # TODO (Q3.0): Add self.refine([dense, rgb]) as a dense residual.
        # Flatten B,V before Conv2d, then restore them.
        residual = self.refine(torch.cat([dense, rgb], 2).flatten(0, 1))
        refined = dense + residual.reshape(b, v, self.channels, h, w)
        # TODO (Q3.0): Project confidence, reconstruct its map, and activate
        # depth/confidence.
        conf = patches_to_image(self.confidence(patches), (h, w), p, 1)
        return {"depth": refined.exp(), "depth_conf": 1 + conf.exp()}


class PointHead(nn.Module):
    """Reserved for Q3.3 (bonus); not part of Q3.0."""

    def __init__(self, width, patch_size):
        super().__init__()
        self.width = width
        self.patch_size = patch_size

    def forward(self, patches, rgb):
        """Reserved interface: (B,V,P,2D), RGB -> (B,V,3,H,W) world points."""
        raise NotImplementedError("Q3.3: PointHead is reserved for the bonus question")
