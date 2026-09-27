"""Minimal decoder-only transformer with selectable positional encoding.

PEs: nope | sinusoidal | learned-abs | rope | alibi
The model predicts, at each query position i, the token s[t(i)] (see src/data.py); this matches
the positional-kernel abstraction in theory/theory.md so empirical LG is directly comparable to alpha_L.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_table(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def alibi_slopes(h):
    # standard ALiBi geometric slopes
    start = 2 ** (-8 / h)
    return torch.tensor([start ** (i + 1) for i in range(h)])


def apply_rope(x, seq, scale=1.0):
    # x: (B, H, T, hd).  rotary on head dim.  scale<1 shrinks all frequencies (low-freq RoPE:
    # smoother kernel that cannot sharply resolve small offsets -- still a *relative* PE).
    B, H, T, hd = x.shape
    half = hd // 2
    freqs = scale * torch.exp(torch.arange(0, half, device=x.device).float() * (-math.log(10000.0) / half))
    ang = torch.arange(T, device=x.device).float().unsqueeze(1) * freqs.unsqueeze(0)  # T, half
    cos = torch.cos(ang)[None, None]  # 1,1,T,half
    sin = torch.sin(ang)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


ROPE_SCALE = {"rope": 1.0, "rope-lowfreq": 0.08}  # both relative; differ only in frequency band


class Attn(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__()
        self.h, self.hd, self.pe = h, d // h, pe
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        if pe == "alibi":
            self.register_buffer("slopes", alibi_slopes(h))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        if self.pe.startswith("rope"):
            sc = ROPE_SCALE.get(self.pe, 1.0)
            q, k = apply_rope(q, T, sc), apply_rope(k, T, sc)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)  # B,H,T,T
        if self.pe == "alibi":
            pos = torch.arange(T, device=x.device)
            rel = (pos[None, :] - pos[:, None]).clamp(max=0).float()  # j-i <=0
            att = att + self.slopes[None, :, None, None] * rel[None, None]
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf"))
        out = F.softmax(att, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h, pe)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Tiny(nn.Module):
    def __init__(self, vocab, d=64, h=4, layers=2, pe="rope", max_train_len=24):
        super().__init__()
        self.pe = pe
        self.tok = nn.Embedding(vocab, d)
        if pe == "sinusoidal":
            self.register_buffer("sin", sinusoidal_table(4096, d))
        if pe == "learned-abs":
            self.posemb = nn.Embedding(max_train_len, d)
            self.max_train_len = max_train_len
        self.blocks = nn.ModuleList([Block(d, h, pe) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pe == "sinusoidal":
            x = x + self.sin[:T].to(x.device)
        elif self.pe == "learned-abs":
            pos = torch.arange(T, device=idx.device).clamp(max=self.max_train_len - 1)  # >=L0 undefined -> clamp
            x = x + self.posemb(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))
