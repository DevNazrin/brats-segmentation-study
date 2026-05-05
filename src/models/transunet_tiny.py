"""
TransUNet-Tiny: a lightweight Transformer-based 3D segmentation model.

Designed for low-resource settings (single consumer GPU, modest training time).
The architecture is intentionally smaller than the original TransUNet to fit on
a Kaggle T4 GPU while still demonstrating Transformer-style global context modeling.

Architecture overview:
  Input (B, 4, D, H, W)
    → Patch embedding (3D conv with stride = patch_size)
    → Transformer encoder (multi-head self-attention + MLP, repeated)
    → Reshape back to spatial 3D feature map
    → 3 transposed-conv upsampling stages
    → 1x1x1 conv to output segmentation logits
"""

import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """
    Splits a 3D volume into non-overlapping patches and projects each patch
    into a fixed-size embedding vector.

    Implemented as a Conv3d with kernel_size = stride = patch_size, which
    is the standard trick for patch embedding.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D/p, H/p, W/p)
        # Flatten spatial dims into a sequence of tokens
        B, E, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, embed_dim)
        return x, (D, H, W)


class TransformerEncoderBlock(nn.Module):
    """One Transformer block: multi-head self-attention + MLP, both with residual + LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNetTiny(nn.Module):
    """
    Lightweight Transformer-based 3D segmentation network.

    Args:
        in_channels: number of input modalities (4 for BraTS).
        out_channels: number of output classes (2 for binary, more for multi-class).
        embed_dim: Transformer embedding dimension. Smaller = lighter model.
        num_heads: number of attention heads.
        depth: number of Transformer blocks stacked.
        patch_size: side length of cubic patches the volume is split into.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        embed_dim: int = 128,
        num_heads: int = 4,
        depth: int = 4,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. Patch embedding: (B, in_channels, D, H, W) → (B, N, embed_dim)
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size)

        # 2. Transformer encoder: stack of self-attention blocks
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        # 3. Decoder: 3 transposed-conv upsampling stages back to (D, H, W)
        # patch_size=8 means we need to upsample 8x total: 2 * 2 * 2 = 8.
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GroupNorm(4, embed_dim // 8),
            nn.GELU(),
        )

        # 4. Output 1x1x1 conv: per-voxel class logits
        self.head = nn.Conv3d(embed_dim // 8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, D, H, W)
        tokens, (Dp, Hp, Wp) = self.patch_embed(x)  # (B, N, embed_dim)

        tokens = self.encoder(tokens)  # (B, N, embed_dim)

        # Reshape token sequence back to a 3D feature map
        B, N, E = tokens.shape
        feat = tokens.transpose(1, 2).reshape(B, E, Dp, Hp, Wp)

        feat = self.decoder(feat)  # upsample to (B, embed_dim/8, D, H, W)
        logits = self.head(feat)   # (B, out_channels, D, H, W)
        return logits


def build_transunet_tiny(in_channels: int = 4, out_channels: int = 2) -> TransUNetTiny:
    """
    Convenience builder. Default settings match the first-semester baseline
    (embed_dim=128, num_heads=4, depth=4, patch_size=8).
    """
    return TransUNetTiny(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=128,
        num_heads=4,
        depth=4,
        patch_size=8,
    )