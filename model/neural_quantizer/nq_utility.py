import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def cal_correlation(x, y):
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    denominator1 = torch.sqrt(torch.sum(x_centered ** 2, dim=1))
    denominator2 = torch.sqrt(torch.sum(y_centered ** 2, dim=1))
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = denominator1 * denominator2
    correlations = numerator / denominator
    correlations[denominator == 0] = 0
    return correlations.mean()

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    confidences, preds = probs.max(dim=1)
    accuracies = (preds == labels).float()
    ece = probs.new_tensor(0.0)
    bin_bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i + 1]
        mask = (confidences > lo) & (confidences <= hi if i < n_bins-1 else confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece = ece + mask.float().mean() * (bin_conf - bin_acc).abs()
    return ece

def brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return ((probs - one_hot) ** 2).sum(dim=-1).mean()

def entropy_rate_from_logits(bigram_logits: torch.Tensor, power_iters: int = 50):
    P = torch.softmax(bigram_logits, dim=-1)
    K = P.size(0)
    if K < 2:
        H_rows = -(P.clamp_min(1e-12) * torch.log2(P.clamp_min(1e-12))).sum(dim=-1)
        return H_rows.mean(), 0.0, 1.0
    pi = torch.ones(K, device=P.device) / K
    for _ in range(power_iters): pi = pi @ P
    H_rows = -(P.clamp_min(1e-12) * torch.log2(P.clamp_min(1e-12))).sum(dim=-1)
    H_rate = (pi * H_rows).sum()
    evals = torch.linalg.eigvals(P)
    lambda2 = evals.abs().topk(2).values[-1].item()
    gap = 1.0 - float(lambda2)
    return H_rate, float(lambda2), float(gap)

def average_run_length(idx: torch.Tensor) -> float:
    B, T = idx.shape
    if T == 0: return .0
    total_runs = 0
    total_len = 0
    for b in range(B):
        x = idx[b]
        starts = torch.cat([torch.tensor([0], device=idx.device), torch.nonzero(x[1:] != x[:-1], as_tuple=False).flatten()+1])
        ends = torch.cat([starts[1:], torch.tensor([T], device=idx.device)])
        run_lens = (ends - starts).float()
        total_runs += run_lens.numel()
        total_len += run_lens.sum().item()
    return total_len / max(1, total_runs)

def mi_lag_k(idx: torch.Tensor, K: int, lag: int = 2) -> float:
    if idx.size(1) <= lag: return 0
    x = idx[:, :-lag].reshape(-1)
    y = idx[:, lag:].reshape(-1)
    xy = x * K + y
    counts_xy = torch.bincount(xy, minlength=K*K).float()
    counts_x = torch.bincount(x, minlength=K).float()
    counts_y = torch.bincount(y, minlength=K).float()
    N = counts_xy.sum().clamp_min(1.0)
    pxy = counts_xy / N
    px = counts_x / N
    py = counts_y / N
    pxpy = (px.unsqueeze(1) * py.unsqueeze(0)).reshape(-1)
    mask = (pxy > 0) & (pxpy > 0)
    mi = (pxy[mask] * (torch.log2(pxy[mask]) - torch.log2(pxpy[mask]))).sum()
    return float(mi.item())

def effective_k_fraction(H_bits: torch.Tensor, K: int) -> float:
    return float((torch.exp2(H_bits) / K).item())

def trigram_cond_entropy_bits_hard(idx: torch.Tensor, K: int, sample_max: int = 200_000) -> float:
    if idx.size(1) < 3: return .0
    device = idx.device
    ctx = idx[:, :-2]
    mid = idx[:, 1:-1]
    nxt = idx[:, 2:]
    B, Tm = mid.shape
    N = B * Tm
    sel = torch.randperm(N, device=device)[:min(N, sample_max)]
    c = (ctx.reshape(-1)[sel] * K + mid.reshape(-1)[sel])          # [S]
    y = nxt.reshape(-1)[sel]                                       # [S]
    pair_id = c * K + y                                            # [S]
    uniq_pair, counts_pair = torch.unique(pair_id, return_counts=True)   # [P]
    uniq_ctx = torch.div(uniq_pair, K, rounding_mode='floor')            # [P]
    uniq_c, inv = torch.unique(uniq_ctx, return_inverse=True)            # [C], [P]
    tot_per_ctx = torch.zeros(uniq_c.size(0), device=device, dtype=counts_pair.dtype)
    tot_per_ctx.scatter_add_(0, inv, counts_pair)
    counts_pair_f = counts_pair.to(torch.float32)
    S1 = (counts_pair_f * torch.log2(counts_pair_f)).sum()
    tot_per_ctx_f = tot_per_ctx.to(torch.float32).clamp_min(1)
    S2 = (tot_per_ctx_f * torch.log2(tot_per_ctx_f)).sum()
    Ntot = counts_pair_f.sum().clamp_min(1.0)
    H_bits = (S2 - S1) / Ntot
    return float(H_bits.item())
