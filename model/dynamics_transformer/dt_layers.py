import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AxialARCFG:
    vocab: int
    n_neurons: int
    n_sessions: int = 3
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.1
    attn_dropout: float = 0.1
    rope_theta: float = 10000.0
    use_abs_time_emb: bool = False
    neuron_topk: int = -1
    neuron_subsample_ratio: float = 0.5
    neuron_subsample_min: int = 24
    train_window_T: int = 0
    eval_use_full_trial: bool = True
    
class LayerNormFP32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.dtype
        return super().forward(x.float()).to(t)

class TemporalMHA_RoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, rope_theta=10000.0, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        assert self.dh % 2 == 0, "RoPE head dim must be even."
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(attn_dropout)
        self.causal = causal
        inv = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _apply_rope(self, q, k, time_pos):
        freqs = torch.einsum('bt,d->btd', time_pos.float(), self.inv_freq)  # [B,T,Dh/2]
        cos = torch.cos(freqs).unsqueeze(1)  # [B,1,T,Dh/2]
        sin = torch.sin(freqs).unsqueeze(1)
        cos = cos.repeat_interleave(2, dim=-1)  # [B,1,T,Dh]
        sin = sin.repeat_interleave(2, dim=-1)
        def rotate_half(x):
            x_even = x[..., ::2]
            x_odd  = x[..., 1::2]
            return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

    def forward(self, x, time_pos, key_padding_mask=None):
        B_, T, D = x.shape
        qkv = self.qkv(x).reshape(B_, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # [3,B,H,T,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,Dh]
        q, k = self._apply_rope(q, k, time_pos)
        attn_mask = None
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool)  # True=valid
            attn_mask = torch.zeros((B_, 1, 1, T), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill((~kpm)[:, None, None, :], float("-inf"))
        dropout_p = self.drop.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal,
        )  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().reshape(B_, T, D)
        return self.proj(y)

class NeuronMHA(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, k_keep: int = -1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.k_keep = k_keep
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B_, N, D = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # [3,B,H,N,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,Dh]
        if self.k_keep is not None and self.k_keep > 0 and self.k_keep < N:
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B,H,N,N]
            topv, topi = torch.topk(att, self.k_keep, dim=-1)
            mask = torch.full_like(att, float('-inf'))
            att = mask.scatter(-1, topi, topv)
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            y = torch.matmul(att, v)  # [B,H,N,Dh]
        else:
            dropout_p = self.drop.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )  # [B,H,N,Dh]
        y = y.transpose(1, 2).contiguous().reshape(B_, N, D)
        return self.proj(y)

class AxialARBlock(nn.Module):
    def __init__(self, cfg: AxialARCFG):
        super().__init__()
        D = cfg.d_model
        self.t_ln = LayerNormFP32(D)
        self.t_attn = TemporalMHA_RoPE(D, cfg.n_heads, cfg.attn_dropout, rope_theta=cfg.rope_theta, causal=True)
        self.n_ln = LayerNormFP32(D)
        self.n_attn = NeuronMHA(D, cfg.n_heads, cfg.attn_dropout, k_keep=cfg.neuron_topk if cfg.neuron_topk>0 else -1)
        self.ff_ln = LayerNormFP32(D)
        self.ff = nn.Sequential(nn.Linear(D, cfg.d_ff, bias=False), nn.GELU(), nn.Linear(cfg.d_ff, D, bias=False))
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("res_scale", torch.tensor(1.0 / math.sqrt(3 * cfg.n_layers), dtype=torch.float32), persistent=False)

    def forward(self, xBNTD, time_posBT, time_kpmaskBT):
        B, N, T, D = xBNTD.shape
        x = xBNTD.reshape(B*N, T, D)
        tp = time_posBT.unsqueeze(1).expand(B, N, T).contiguous().reshape(B*N, T)
        tk = time_kpmaskBT.unsqueeze(1).expand(B, N, T).contiguous().reshape(B*N, T)
        x = x + self.res_scale * self.drop(self.t_attn(self.t_ln(x), tp, key_padding_mask=tk))
        x = x.reshape(B, N, T, D)
        pad_mask = time_kpmaskBT.unsqueeze(1).unsqueeze(-1).to(x.dtype)
        x = x * pad_mask
        x2 = x.permute(0,2,1,3).contiguous().reshape(B*T, N, D)
        x2 = x2 + self.res_scale * self.drop(self.n_attn(self.n_ln(x2)))
        x = x2.reshape(B, T, N, D).permute(0,2,1,3).contiguous()
        x = x + self.res_scale * self.drop(self.ff(self.ff_ln(x)))
        return x
    