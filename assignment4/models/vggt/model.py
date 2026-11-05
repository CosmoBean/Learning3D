"""Q3.0: Mini-VGGT; complete attention aggregation and prediction heads.

B: batch size; V: views; P: image patches; N=P+5: all tokens; D: model width.
Encoder, token setup, camera decoding and model wiring are provided.
"""

import torch
from torch import nn

from models.vggt.attention import GeometryBlock
from models.vggt.camera import decode_camera
from models.vggt.encoder import FrozenDINO
from models.vggt.heads import CameraHead, DepthHead, PointHead


class MiniVGGT(nn.Module):
    def __init__(
        self,
        image_size=224,
        embed_dim=256,
        num_heads=4,
        num_layers=8,
        encoder_repo=None,
        encoder_weights=None,
        encoder_pretrained=True,
        predict_points=False,
        encoder=None,
        encoder_name="dinov2_vits14",
    ):
        """Provided: initialize encoder, tokens, blocks and prediction heads.

        num_layers counts frame/global pairs (8 pairs = 16 distinct blocks).
        encoder is an optional stand-in for small offline checks.
        The camera transformer is provided; complete only its final prediction.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("MiniVGGT requires at least one frame/global pair")
        if encoder is None:
            options = {
                "weights": encoder_weights,
                "pretrained": encoder_pretrained,
                "name": encoder_name,
            }
            if encoder_repo is not None:
                options["repo"] = encoder_repo
            encoder = FrozenDINO(**options)
        # Q3.3: also enable gradients here when fine-tuning; see FrozenDINO.
        self.encoder = encoder.requires_grad_(False).eval()
        self.image_size, self.patch_size = image_size, encoder.patch_size
        if image_size % self.patch_size:
            raise ValueError("image_size must be divisible by the DINO patch size")
        self.grid = image_size // self.patch_size
        self.patch_start = 5  # One camera token and four register tokens.
        self.projection = nn.Linear(encoder.embed_dim, embed_dim)
        # Distinguish camera 0; remaining views share the same special tokens.
        self.register_tokens = nn.Parameter(torch.randn(1, 2, 4, embed_dim) * 0.02)
        y, x = torch.meshgrid(
            torch.arange(self.grid) + 1, torch.arange(self.grid) + 1, indexing="ij"
        )
        positions = torch.cat(
            [
                torch.zeros(self.patch_start, 2),
                torch.stack([y, x], -1).reshape(-1, 2),
            ]
        )
        self.register_buffer("positions", positions, persistent=False)
        self.camera_tokens = nn.Parameter(torch.randn(1, 2, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [GeometryBlock(embed_dim, num_heads) for _ in range(2 * num_layers)]
        )
        head_dim = 2 * embed_dim
        self.norm = nn.LayerNorm(head_dim)
        self.camera_head = CameraHead(head_dim, num_heads)
        self.depth_head = DepthHead(head_dim, self.patch_size)
        # Reserved Q3.3 bonus interface; disabled for Q3.0–3.2.
        self.point_head = PointHead(head_dim, self.patch_size) if predict_points else None

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def aggregate(self, tokens):
        """Q3.0: Alternate frame/global blocks, then concatenate final-pair features.

        Args:
            tokens: (B,V,N,D), camera/register/patch tokens.
            self.positions: (N,2), per-view 2D token positions.

        Returns:
            tokens: (B,V,N,2D), final frame/global outputs concatenated along
                the feature dimension and passed through self.norm.

        Even-indexed blocks attend within each view: (B*V,N,D).
        Odd-indexed blocks attend across all views: (B,V*N,D).
        Repeat positions for each view in global attention, without adding a
        temporal/view-order positional embedding; camera 0 is already marked.
        Save the final frame block's output before its global block updates it.
        """
        b, v, n, d = tokens.shape
        global_positions = self.positions.repeat(v, 1)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # TODO (Q3.0): Frame attention; reshape to (B*V,N,D), call
                # block with self.positions, restore (B,V,N,D), and save
                # this frame output for the final concatenation.
                tokens = block(tokens.reshape(b * v, n, d), self.positions)
                tokens = frame = tokens.reshape(b, v, n, d)
            else:
                # TODO (Q3.0): Global attention; reshape to (B,V*N,D), call
                # block with positions repeated per view, restore (B,V,N,D).
                tokens = block(tokens.reshape(b, v * n, d), global_positions)
                tokens = tokens.reshape(b, v, n, d)
        # TODO (Q3.0): Concatenate the final frame/global outputs along D
        # and return self.norm(...) with shape (B,V,N,2D).
        return self.norm(torch.cat([frame, tokens], dim=-1))

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Provided wiring: RGB-only inputs -> supervised cameras and dense depth.

        Args:
            inputs: vggt_inputs(batch), rgb (B,V,3,H,W) in [0,1].

        Returns:
            K: (B,V,3,3), intrinsics in pixels.
            T_world_from_camera: (B,V,4,4), predicted c2w poses.
            depth: (B,V,1,H,W), positive normalized optical-axis z-depth.
            depth_conf: (B,V,1,H,W), confidence >=1.
            pose_enc: (B,V,9), raw activated encoding for direct camera loss.
        """
        rgb = inputs["rgb"]
        b, v, _, h, w = rgb.shape
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError("Input size must match model image_size")
        # Q3.3: remove this block and FrozenDINO.forward's decorator to fine-tune.
        with torch.no_grad():
            features = self.encoder(rgb.flatten(0, 1))  # (B*V,P,self.encoder.embed_dim).
        patches = self.projection(features).reshape(b, v, self.grid**2, -1)
        camera_tokens = torch.cat(
            [
                self.camera_tokens[:, :1],
                self.camera_tokens[:, 1:].expand(1, v - 1, 1, -1),
            ],
            dim=1,
        ).expand(b, -1, -1, -1)
        registers = torch.cat(
            [
                self.register_tokens[:, :1],
                self.register_tokens[:, 1:].expand(1, v - 1, 4, -1),
            ],
            dim=1,
        ).expand(b, -1, -1, -1)
        tokens = self.aggregate(torch.cat([camera_tokens, registers, patches], dim=2))
        camera = self.camera_head(tokens[:, :, 0]).float()  # (B,V,2D) -> (B,V,9).
        poses, K = decode_camera(camera, (h, w))
        dense = self.depth_head(tokens[:, :, self.patch_start :], rgb)
        if self.point_head is not None:
            dense["points_world"] = self.point_head(tokens[:, :, self.patch_start :], rgb)
        return {
            "K": K,
            "T_world_from_camera": poses,
            "pose_enc": camera,
            **dense,
        }
