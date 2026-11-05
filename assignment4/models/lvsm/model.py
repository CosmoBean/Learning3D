"""Mini-LVSM: a decoder-only transformer that renders a target view from tokens.

Source images (RGB + Plucker rays) and target-camera queries (Plucker rays only)
are patchified into tokens, concatenated into one sequence, processed by a
standard transformer, and the output at the target-query tokens is decoded back
into RGB patches. No 3D representation (NeRF, Gaussians, point cloud) is built.

Layout of this file (Part 2 TODOs are marked `# TODO (Q2.x)`; the rest is provided):

    provided : sincos_2d_pos_embed, TransformerBlock, MiniLVSM layers
    TODO     : patchify, unpatchify                     (Q2.0)
    TODO     : MiniLVSM.build_tokens                    (Q2.0)
    TODO     : MiniLVSM.forward                         (Q2.0)

Symbols: B batch, S source views, T target views, H/W image size, p patch size,
N = (H/p)*(W/p) patches per view, D transformer width.
"""

# ruff: noqa: F841

import math

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# Provided utilities: positional encoding and transformer blocks
# --------------------------------------------------------------------------- #


def sincos_2d_pos_embed(dim, grid_h, grid_w, device=None, dtype=torch.float32):
    """Fixed 2D sine-cosine embedding for a (grid_h, grid_w) patch grid.

    Returns:
        (grid_h * grid_w, dim) in row-major patch order (same order as patchify).
    """
    if dim % 4 != 0:
        raise ValueError("Embedding dim must be divisible by 4")
    quarter = dim // 4
    freqs = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )

    def encode(pos):
        angles = pos.reshape(-1, 1) * freqs[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    embed = torch.cat([encode(ys), encode(xs)], dim=-1)
    return embed.to(device=device, dtype=dtype)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, head_dim=64, mlp_ratio=4.0, use_qk_norm=True):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError("dim must be divisible by head_dim")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if use_qk_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)  # full (bidirectional) attention
        x = x + self.proj(out.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------- #
# Q2.0  Patchify / unpatchify
# --------------------------------------------------------------------------- #


def patchify(images, patch_size):
    """Q2.0: Split images into flattened non-overlapping patches.

    Args:
        images: (..., C, H, W) with H and W divisible by patch_size. Any number
            of leading dims is allowed, e.g. (B, V, C, H, W).
        patch_size: p.

    Returns:
        (..., N, p*p*C) with N = (H/p)*(W/p). Patches are in row-major order
        (left to right, then top to bottom); each patch is flattened in
        (row, col, channel) order, so unpatchify is its exact inverse.
    """
    *lead, c, h, w = images.shape
    p = patch_size
    if h % p or w % p:
        raise ValueError(f"Image size {(h, w)} not divisible by patch size {p}")
    gh, gw = h // p, w // p
    # TODO (Q2.0): Flatten all leading dims into one and split H and W into
    # (gh, p) and (gw, p) blocks, giving (L, C, gh, p, gw, p).
    x = images.reshape(-1, c, gh, p, gw, p)
    # TODO (Q2.0): Reorder to (L, gh, gw, p, p, C): row-major patches, each
    # flattened in (row, col, channel) order.
    x = x.permute(0, 2, 4, 3, 5, 1)
    # TODO (Q2.0): Merge the grid and each patch's contents, restoring the
    # leading dims: (*lead, gh*gw, p*p*C).
    return x.reshape(*lead, gh * gw, p * p * c)


def unpatchify(patches, patch_size, image_hw):
    """Q2.0: Inverse of `patchify`.

    Args:
        patches: (..., N, p*p*C) with N = (H/p)*(W/p).
        patch_size: p.
        image_hw: (H, W) of the image to reconstruct.

    Returns:
        (..., C, H, W).
    """
    *lead, n, dim = patches.shape
    p = patch_size
    h, w = image_hw
    gh, gw = h // p, w // p
    if n != gh * gw or dim % (p * p) != 0:
        raise ValueError("Patch tensor does not match patch_size / image_hw")
    c = dim // (p * p)
    # TODO (Q2.0): Flatten leading dims and unflatten each patch to
    # (L, gh, gw, p, p, C).
    x = patches.reshape(-1, gh, gw, p, p, c)
    # TODO (Q2.0): Reorder to (L, C, gh, p, gw, p).
    x = x.permute(0, 5, 1, 3, 2, 4)
    # TODO (Q2.0): Merge to (*lead, C, H, W).
    return x.reshape(*lead, c, h, w)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class MiniLVSM(nn.Module):
    """Decoder-only LVSM.

    Each target view is rendered independently: its query tokens are appended to
    the tokens of all source views and the whole sequence attends to itself.
    Target views never see each other or any target RGB.

    Args:
        patch_size: p; image H and W must be divisible by it.
        d_model: Token width D (divisible by 4 and by head_dim).
        n_layers: Number of transformer blocks.
        head_dim: Width of each attention head.
        mlp_ratio: MLP hidden width / D.
        use_qk_norm: Normalize queries/keys per head (stabilizes small-model training).
    """

    def __init__(
        self,
        patch_size=16,
        d_model=384,
        n_layers=8,
        head_dim=64,
        mlp_ratio=4.0,
        use_qk_norm=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        p2 = patch_size**2
        # Provided: token projections. Source patch = RGB(3) + Plucker(6) channels,
        # target patch = Plucker(6) channels only (no target RGB is ever seen).
        # The two separate projections (each with its own bias) already tell the
        # transformer which tokens are sources and which are target queries.
        self.source_proj = nn.Linear(p2 * 9, d_model)
        self.target_proj = nn.Linear(p2 * 6, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, head_dim, mlp_ratio, use_qk_norm)
            for _ in range(n_layers)
        )
        # Provided: RGB head, decodes each target token into a p*p*3 patch.
        self.out_norm = nn.LayerNorm(d_model)
        self.rgb_head = nn.Linear(d_model, p2 * 3)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---- Q2.0 ------------------------------------------------------------ #
    def build_tokens(self, source_tokens, target_tokens):
        """Q2.0: Assemble one transformer sequence per target view.

        Args:
            source_tokens: (B, S, N, D) projected source tokens (RGB + rays).
            target_tokens: (B, T, N, D) projected target-query tokens (rays only).

        Returns:
            (B*T, S*N + N, D): for every (sample, target) pair, all source tokens
            followed by that target's N query tokens. Positional embedding (shared
            by sources and targets, depends only on the patch location) has been
            added. Row b*T + t is sample b, target t.
        """
        b, s, n, d = source_tokens.shape
        t = target_tokens.shape[1]
        grid = int(math.isqrt(n))
        # TODO (Q2.0): Build the (N, D) positional embedding for the grid x grid
        # patch layout with sincos_2d_pos_embed, matching the tokens' device/dtype.
        pos = sincos_2d_pos_embed(
            d, grid, grid, device=source_tokens.device, dtype=source_tokens.dtype
        )
        # TODO (Q2.0): Add it to the source and target tokens.
        source = source_tokens + pos  # (B, S, N, D)
        target = target_tokens + pos  # (B, T, N, D)
        # TODO (Q2.0): Flatten each sample's sources into S*N tokens and repeat
        # them for every target: (B, T, S*N, D).
        source = source.reshape(b, 1, s * n, d).expand(b, t, s * n, d)
        # TODO (Q2.0): Append each target's N query tokens after its sources,
        # then fold T into the batch: (B*T, (S+1)*N, D).
        return torch.cat([source, target], dim=2).reshape(b * t, (s + 1) * n, d)

    # ---- Q2.0 ------------------------------------------------------------ #
    def forward(self, inputs):
        """Q2.0: Synthesize target RGB without receiving target RGB as input.

        Args:
            inputs: lvsm_inputs(batch, num_source_views), containing:
                source_rgb: (B, S, 3, H, W), RGB in [0, 1].
                source_plucker: (B, S, 6, H, W), normalized [direction, moment].
                target_plucker: (B, T, 6, H, W), normalized [direction, moment].

        Returns:
            Dict with rgb: (B, T, 3, H, W), target RGB in [0, 1], in target view order.
        """
        source_rgb = inputs["source_rgb"]
        target_plucker = inputs["target_plucker"]
        b, t = target_plucker.shape[:2]
        h, w = source_rgb.shape[-2:]
        p = self.patch_size
        n = (h // p) * (w // p)

        # TODO (Q2.0): Concatenate source RGB shifted to [-1, 1] with the source
        # Plucker rays along channels: (B, S, 9, H, W).
        source = torch.cat([source_rgb * 2 - 1, inputs["source_plucker"]], dim=2)
        # TODO (Q2.0): Patchify and project the sources with self.source_proj
        # and the target rays with self.target_proj (the target never sees RGB).
        source_tokens = self.source_proj(patchify(source, p))  # (B, S, N, D)
        target_tokens = self.target_proj(patchify(target_plucker, p))  # (B, T, N, D)
        # TODO (Q2.0): Build the tokens, apply self.input_norm, then every block.
        tokens = self.input_norm(self.build_tokens(source_tokens, target_tokens))
        for block in self.blocks:
            tokens = block(tokens)  # (B*T, (S+1)*N, D)
        # TODO (Q2.0): Keep only each sequence's last N (target-query) tokens and
        # decode them with self.out_norm, self.rgb_head and a sigmoid.
        patches = torch.sigmoid(self.rgb_head(self.out_norm(tokens[:, -n:])))
        # TODO (Q2.0): Unpatchify and restore (B, T, 3, H, W).
        rgb = unpatchify(patches, p, (h, w)).reshape(b, t, 3, h, w)
        return {"rgb": rgb}
