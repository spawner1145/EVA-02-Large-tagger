import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=True):
        super().__init__()
        self.eps = eps
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * (1 + self.weight)
        if self.use_bias:
            x = x + self.bias
        return x

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, image_size=224, patch_size=16, theta=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension {dim} must be even for rotary embedding")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.dim = dim
        seq_len = (image_size // patch_size) ** 2
        exp = torch.arange(0, dim, 2, dtype=torch.float32) / -dim
        freqs = theta ** exp
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, freqs)
        freqs = freqs.repeat(1, 2).view(-1, dim)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x = torch.stack([-x2, x1], dim=-1)
        return x.view(*x.shape[:-2], -1)

    def forward(self, x):
        assert x.shape[-1] == self.dim, f"Expected dimension {self.dim}, got {x.shape[-1]}"
        return x * self.freqs_cos + self.rotate_half(x) * self.freqs_sin

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_extra_tokens, use_bias=True, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Dimension {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.num_extra_tokens = num_extra_tokens
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=use_bias)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.rope = VisionRotaryEmbedding(self.head_dim, image_size=224, patch_size=16)  # 修复点

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        assert q.shape == (B, self.num_heads, N, self.head_dim), f"Q shape mismatch: {q.shape}"

        q_xtr, q_seq = q[:, :, :self.num_extra_tokens], q[:, :, self.num_extra_tokens:]
        k_xtr, k_seq = k[:, :, :self.num_extra_tokens], k[:, :, self.num_extra_tokens:]
        q_seq = self.rope(q_seq)
        k_seq = self.rope(k_seq)
        q = torch.cat([q_xtr, q_seq], dim=2)
        k = torch.cat([k_xtr, k_seq], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) if self.training else attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x) if self.training else x
        return x

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_features, use_bias=True, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_features * 2, bias=use_bias)
        self.fc2 = nn.Linear(hidden_features, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout(x) if self.training else x
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, use_bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=use_bias)

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError(f"Image dimensions {x.shape[2:]} must be divisible by patch_size {self.patch_size}")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class EVA02TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, num_extra_tokens, drop_path=0.0, use_bias=True):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, num_extra_tokens, use_bias=use_bias)
        self.norm2 = LayerNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim, use_bias=use_bias)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EVA02Transformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, num_layers=12, embed_dim=768,
                 mlp_dim=3072, num_heads=12, drop_path_rate=0.1, dropout=0.0, use_bias=True):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, use_bias=use_bias)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.num_extra_tokens = 1

        dpr = np.linspace(0, drop_path_rate, num_layers).tolist()
        self.blocks = nn.ModuleList([
            EVA02TransformerBlock(embed_dim, num_heads, mlp_dim, self.num_extra_tokens, dpr[i], use_bias)
            for i in range(num_layers)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=use_bias) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        assert x.shape == (B, (self.pos_embed.shape[1] - 1), self.pos_embed.shape[2]), f"Patch embed shape mismatch: {x.shape}"
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, self.num_extra_tokens:].mean(dim=1))
        x = self.head(x)
        return x

    @classmethod
    def build(cls, config, num_classes, **kwargs):
        config_dict = {k: kwargs.get(k, v) for k, v in vars(config).items()}
        config_dict["num_classes"] = num_classes
        return cls(**config_dict)

def eva02_large():
    class Config:
        image_size = 224
        patch_size = 16
        num_layers = 24
        embed_dim = 1024
        mlp_dim = (1024 * 4 * 2) // 3
        num_heads = 16
        drop_path_rate = 0.1
        use_bias = True
        scale_mlp = True

        def build(self, **kwargs):
            return EVA02Transformer.build(self, **kwargs)

    return Config()