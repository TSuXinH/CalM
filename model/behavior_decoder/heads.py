from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodeHeadLowRankUV(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_neurons_total: int,
        beh_channels: int = 3,
        up: int = 4,
        rank: int = 32,
        conv_kernel: int = 9,
        conv_init_identity: bool = True,
        conv_bias: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.C = int(beh_channels)
        self.up = int(up)
        self.O = int(self.C * self.up)
        self.Ntot = int(n_neurons_total)
        D = int(d_model)
        R = int(rank)
        self.rank = R
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.A_u = nn.Parameter(torch.randn(self.O, R, D) * 0.02)
        self.A_v = nn.Parameter(torch.randn(self.O, R, D) * 0.02)
        self.B = nn.Parameter(torch.randn(self.O, self.Ntot, R) * 0.02)
        self.b = nn.Parameter(torch.zeros(self.O))
        k = int(conv_kernel)
        if k > 1:
            if k % 2 == 0:
                raise ValueError('conv_kernel must be odd when smoothing is enabled.')
            self.smooth = nn.Conv1d(self.C, self.C, kernel_size=k, padding=k // 2, groups=self.C, bias=bool(conv_bias))
            if bool(conv_init_identity):
                with torch.no_grad():
                    self.smooth.weight.zero_()
                    self.smooth.weight[:, 0, k // 2] = 1.0
                    if self.smooth.bias is not None:
                        self.smooth.bias.zero_()
        else:
            self.smooth = None

    def forward(self, hBNTD: torch.Tensor, neuron_idsBN: torch.Tensor) -> torch.Tensor:
        B, N, T, _ = hBNTD.shape
        nid = neuron_idsBN
        if int(nid.min().item()) < 0 or int(nid.max().item()) >= self.Ntot:
            raise ValueError(f'neuron_ids out of range: min={int(nid.min().item())} max={int(nid.max().item())} Ntot={self.Ntot}')
        u = torch.einsum('bntd,ord->bntor', hBNTD, self.A_u)
        v = torch.einsum('bntd,ord->bntor', hBNTD, self.A_v)
        z = self.drop(u * F.gelu(v))
        Bsel = self.B[:, nid, :].permute(1, 0, 2, 3).contiguous()
        y = torch.einsum('bntor,bonr->bto', z, Bsel) + self.b[None, None, :]
        y = y.view(B, T, self.C, self.up).permute(0, 2, 1, 3).contiguous().view(B, self.C, T * self.up)
        if self.smooth is not None:
            y = self.smooth(y)
        return y
