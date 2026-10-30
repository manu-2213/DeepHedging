# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    """RoPE positional embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, q, k):
        B, H, T, D = q.shape
        positions = torch.arange(T, device=q.device).float()
        angles = positions[:, None] / (10000 ** (torch.arange(0, D, 2, device=q.device) / D))
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        rope = torch.zeros_like(q)
        rope[..., 0::2] = cos
        rope[..., 1::2] = sin
        q_rotated = (q * rope) + self._rotate(q) * (1 - rope)
        k_rotated = (k * rope) + self._rotate(k) * (1 - rope)
        return q_rotated, k_rotated

    def _rotate(self, x):
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]
        x_rot[..., 1::2] = x[..., 0::2]
        return x_rot

class CausalSelfAttention(nn.Module):
    """Causal self-attention module."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.rope = RoPE(self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x).reshape(B, T, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rope(q, k)
        attn_mask = torch.tril(torch.ones((T, T), device=x.device)).view(1, 1, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_dropout(self.proj(y))
        return y

class Block(nn.Module):
    """Transformer block."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series prediction.
    
    Args:
        input_dim: Dimension of input features.
        embed_dim: Embedding dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        output_dim: Dimension of the model output.
        dropout: Dropout rate.
    """
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        B, T, D = x.size()
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        predictions = torch.sigmoid(self.head(x))
        # predictions: (B, T, output_dim)
        return predictions
