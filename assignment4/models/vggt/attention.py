"""Provided frame/global attention block with 2D RoPE."""

import torch
import torch.nn.functional as F
from torch import nn


def rotary_2d(features, positions):
    """Provided: rotate (B,heads,N,d) features at (N,2) row/column positions."""
    axis_dim = features.shape[-1] // 2
    frequencies = 100.0 ** (
        -torch.arange(0, axis_dim, 2, device=features.device).float() / axis_dim
    )
    rotated = []
    for axis, values in enumerate(features.chunk(2, dim=-1)):
        angles = positions[:, axis, None].float() * frequencies
        angles = torch.cat([angles, angles], dim=-1).to(features.dtype)
        first, second = values.chunk(2, dim=-1)
        rotated.append(
            values * angles.cos() + torch.cat([-second, first], dim=-1) * angles.sin()
        )
    return torch.cat(rotated, dim=-1)


class GeometryBlock(nn.Module):
    """Provided layers for pre-norm attention + MLP with residual scales."""

    def __init__(self, width, heads):
        super().__init__()
        if width % heads or (width // heads) % 4:
            raise ValueError("2D RoPE requires head dimension divisible by 4")
        self.heads, self.head_dim = heads, width // heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.attention_scale = nn.Parameter(torch.full((width,), 0.01))
        self.mlp_scale = nn.Parameter(torch.full((width,), 0.01))

    def forward(self, tokens, positions):
        """Provided: return updated tokens with the same shape.

        Args:
            tokens: (M,N,D); M=B*V for frame attention or B for global attention.
            positions: (N,2), row/column coordinates; special tokens use (0,0).

        Returns:
            tokens: (M,N,D), after attention and MLP residual updates.
        """
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm_attention(tokens)).reshape(
            batch, count, 3, self.heads, self.head_dim
        )
        q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rotary_2d(self.q_norm(q), positions)
        k = rotary_2d(self.k_norm(k), positions)
        attended = F.scaled_dot_product_attention(
            q.to(value.dtype), k.to(value.dtype), value, dropout_p=0
        )
        attended = attended.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.attention_scale * self.projection(attended)
        return tokens + self.mlp_scale * self.mlp(self.norm_mlp(tokens))
