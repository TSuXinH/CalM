import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, rope_freqs):
    cos, sin = rope_freqs
    cos = cos[None, :, None, :]  # (1,T,1,D/2)
    sin = sin[None, :, None, :]
    q1, q2 = q[..., ::2], q[..., 1::2]  # (B,T,H,D/2)
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_even = q1 * cos - q2 * sin
    q_odd  = q1 * sin + q2 * cos
    k_even = k1 * cos - k2 * sin
    k_odd  = k1 * sin + k2 * cos
    q_rot = torch.stack((q_even, q_odd), dim=-1).flatten(-2)
    k_rot = torch.stack((k_even, k_odd), dim=-1).flatten(-2)
    return q_rot, k_rot

class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, rope_max_len=2048):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(rope_max_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, D/2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x, mask=None, need_weights=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        if T > self.cos_cached.size(0): raise ValueError(f"T={T} exceeds rope_max_len={self.cos_cached.size(0)}")
        cos = self.cos_cached[:T, :].to(device=x.device, dtype=q.dtype)
        sin = self.sin_cached[:T, :].to(device=x.device, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, (cos, sin))
        attn_scores = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.head_dim)  # (B, H, T, T)
        if mask is not None: 
            mask = mask.to(device=attn_scores.device, dtype=attn_scores.dtype)
            attn_scores = attn_scores + mask[None, None, :, :]  # broadcast mask
        attn_probs = F.softmax(attn_scores, dim=-1).to(attn_scores.dtype)
        attn_probs = self.dropout(attn_probs)
        out = torch.einsum("bhtT,bThd->bthd", attn_probs, v)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        return (out, attn_probs) if need_weights else out

class StandardTransformerBlockWithRoPE(nn.Module):
    def __init__(self, d_qk, d_v, heads, d_hidden, dropout_ratio, rope_max_len=2048):
        super().__init__()
        assert d_qk == d_v
        self.attention = MultiheadAttentionWithRoPE(d_qk, heads, dropout=dropout_ratio, rope_max_len=rope_max_len)
        self.ln1 = nn.LayerNorm(d_qk)
        self.ffn = nn.Sequential(nn.Linear(d_v, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_v))
        self.ln2 = nn.LayerNorm(d_v)

    def forward(self, x, mask=None, if_atten_map=False):
        x_pre_ln1 = self.ln1(x)
        if if_atten_map: x_atten, atten_map = self.attention(x_pre_ln1, mask=mask, need_weights=True)
        else: x_atten = self.attention(x_pre_ln1, mask=mask, need_weights=False)
        x = x + x_atten
        x = x + self.ffn(self.ln2(x))
        return (x, atten_map) if if_atten_map else x
    