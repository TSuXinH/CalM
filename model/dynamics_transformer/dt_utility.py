import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import os, math, random, argparse, heapq, csv, json, glob

def _sync_cuda_if_needed(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        
def _strip_prefix_if_present(sd: dict, prefix: str) -> dict:
    if len(sd) > 0 and all(k.startswith(prefix) for k in sd.keys()): return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def normalize_state_dict_keys(sd: dict) -> dict:
    sd = _strip_prefix_if_present(sd, "_orig_mod.")
    sd = _strip_prefix_if_present(sd, "module.")
    return sd

def load_weights_strict(model_raw, sd_or_path, map_location="cpu"):
    import torch
    if isinstance(sd_or_path, str): sd = torch.load(sd_or_path, map_location=map_location)
    else: sd = sd_or_path
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict): sd = sd["state_dict"]
    sd = normalize_state_dict_keys(sd)
    missing, unexpected = model_raw.load_state_dict(sd, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, (missing[:20], unexpected[:20])
    return model_raw

def ece_score(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)
        correct = (pred == targets).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        ece = 0.0
        for i in range(n_bins):
            m = ((conf >= bins[i]) & (conf <= bins[i+1])) if i == 0 else ((conf > bins[i]) & (conf <= bins[i+1]))
            if m.sum() == 0: continue
            acc = correct[m].mean()
            avg_conf = conf[m].mean()
            ece += (m.float().mean() * (avg_conf - acc).abs()).item()
        return float(ece)
    
def topk_acc_from_logits(logits: torch.Tensor, targets: torch.Tensor, ks=(1,5,10)) -> dict:
    with torch.no_grad():
        max_k = max(ks)
        topk = logits.topk(k=max_k, dim=-1).indices
        out = {}
        for k in ks:
            correct = (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            out[f'acc@{k}'] = correct
        return out
                
@torch.no_grad()
def _maybe_strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith('module.') for k in sd.keys()): return {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd

def load_ground_truth_npz(npz_path: str):
    with np.load(npz_path, allow_pickle=False) as f: X = f["val_padded_X"]; L = f["val_lengths"]
    return np.asarray(X, dtype=np.float32), np.asarray(L).astype(np.int64)

def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2: return np.nan
    if not (np.isfinite(x).all() and np.isfinite(y).all()): return np.nan
    vx, vy = float(np.var(x)), float(np.var(y))
    if vx == 0.0 or vy == 0.0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

@torch.no_grad()
def _pearson_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    vx = x - x.mean(dim=-1, keepdim=True)
    vy = y - y.mean(dim=-1, keepdim=True)
    num = (vx * vy).sum(dim=-1)
    den = (vx.square().sum(dim=-1).sqrt() * vy.square().sum(dim=-1).sqrt()).clamp_min(eps)
    return num / den

def _align_gt_to_stride_mod4_like_training(gt_raw: np.ndarray, stride: int) -> np.ndarray:
    assert gt_raw.ndim == 2
    stride = int(stride)
    if stride != 4: raise ValueError(f"This alignment is defined for stride=4 (your pipeline). Got stride={stride}.")
    N, Ttrue = gt_raw.shape
    if Ttrue <= 0: return gt_raw
    r = Ttrue % 4
    if r == 1: return gt_raw[:, :-1] if Ttrue >= 1 else gt_raw
    elif r == 2:
        last = gt_raw[:, -1:] 
        return np.concatenate([gt_raw, np.repeat(last, repeats=2, axis=1)], axis=1)
    elif r == 3:
        last = gt_raw[:, -1:]  # [N,1]
        return np.concatenate([gt_raw, last], axis=1)
    else: return gt_raw
    
def _unwrap_object(v):
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == (): return v.item()
    return v

def _as_trial_list(container):
    container = _unwrap_object(container)
    if isinstance(container, dict):
        keys = list(container.keys())
        try: keys_sorted = sorted(keys, key=lambda x: int(x))
        except Exception: keys_sorted = keys
        return [container[k] for k in keys_sorted]
    if isinstance(container, (list, tuple)):
        return list(container)
    if isinstance(container, np.ndarray) and container.dtype == object:
        return [_unwrap_object(container[i]) for i in range(len(container))]
    raise TypeError(f"Unsupported container type: {type(container)}")

def load_npz_splits_tokens(npz_path: str, token_key="token", vocab: int = None):
    out = {"train": [], "val": [], "test": []}
    with np.load(npz_path, allow_pickle=True) as z:
        files = set(z.files)
        if not {"train", "val", "test"}.issubset(files): raise ValueError(f"{npz_path} missing train/val/test keys. found={z.files}")
        for sp in ("train", "val", "test"):
            trials_raw = _as_trial_list(z[sp])
            trials_tok = []
            for t in trials_raw:
                t = _unwrap_object(t)
                arr = t[token_key] if isinstance(t, dict) else t
                arr = np.asarray(arr, dtype=np.int64)
                if arr.ndim != 2: raise ValueError(f"{npz_path}:{sp} trial token must be [N,T], got {arr.shape}")
                if vocab is not None:
                    if arr.min() < 0 or arr.max() >= vocab: raise ValueError(f"{npz_path}:{sp} token out of range: min={arr.min()} max={arr.max()} vocab={vocab}")
                trials_tok.append(arr)
            out[sp] = trials_tok
    return out

def build_session_index(
    data_root: str,
    pattern="*.npz",
    exclude_dir_prefix=("prev_",),
    token_key="token",
    vocab: int = None,
):
    paths = sorted(glob.glob(os.path.join(data_root, "**", pattern), recursive=True))
    npz_list = []
    for p in paths:
        parent = os.path.basename(os.path.dirname(p))
        if any(parent.startswith(pref) for pref in exclude_dir_prefix): continue
        npz_list.append(p)
    if len(npz_list) == 0: raise FileNotFoundError(f"No npz found under {data_root} pattern={pattern}")
    sessions = []
    neuron_offset = 0
    for sid, p in enumerate(npz_list):
        splits = load_npz_splits_tokens(p, token_key=token_key, vocab=vocab)
        probe = (splits["train"][0] if len(splits["train"]) else
                 splits["val"][0]   if len(splits["val"])   else
                 splits["test"][0])
        n_neurons = int(probe.shape[0])
        sessions.append({"path": p, "session_id": sid, "neuron_offset": neuron_offset, "n_neurons": n_neurons,})
        neuron_offset += n_neurons
    meta = {"npz_list": npz_list, "n_sessions": len(sessions), "n_neurons_total": neuron_offset}
    return sessions, meta

def unwrap_compiled(m):
    m = getattr(m, '_orig_mod', m)
    if hasattr(m, 'module'):
        m = m.module
    m = getattr(m, '_orig_mod', m)
    return m

def _stable_int_hash(s: str) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def load_npz_splits_tokens_v2(npz_path: str, token_key: str = "token", vocab: int = None):
    def _is_index_dict(d):
        if not isinstance(d, dict) or len(d) == 0:
            return False
        ks = list(d.keys())
        return all(str(k).isdigit() for k in ks)

    def _sort_keys(ks):
        return sorted(ks, key=lambda x: int(str(x)) if str(x).isdigit() else str(x))

    def _extract_tok(trial_obj, split: str):
        if isinstance(trial_obj, np.ndarray):
            if trial_obj.ndim >= 2:
                return trial_obj
            if trial_obj.shape == () and isinstance(trial_obj.item(), dict):
                return _extract_tok(trial_obj.item(), split)
        if isinstance(trial_obj, dict):
            if token_key in trial_obj:
                return trial_obj[token_key]
            for alt in ("tok", "tokens", "token_ids", "tokNT", "codes", "ids", "k"):
                if alt in trial_obj:
                    return trial_obj[alt]
            raise KeyError(
                f"[load_npz_splits_tokens_v2] token_key='{token_key}' not found in trial dict "
                f"({npz_path} split={split}). keys={list(trial_obj.keys())[:30]}"
            )
        tok = getattr(trial_obj, token_key, None)
        if tok is not None: return tok
        arr = np.asarray(trial_obj)
        if arr.ndim >= 2: return arr
        raise ValueError(
            f"[load_npz_splits_tokens_v2] Cannot extract tokens from type={type(trial_obj)} "
            f"({npz_path} split={split})."
        )
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for split in ("train", "val", "test"):
            if split not in z: continue
            obj = z[split]
            trials = []
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                for item in obj:
                    if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item):
                        for k in _sort_keys(item.keys()):
                            tok = _extract_tok(item[k], split)
                            trials.append(np.asarray(tok, dtype=np.int64))
                    else:
                        tok = _extract_tok(item, split)
                        trials.append(np.asarray(tok, dtype=np.int64))
            elif isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
                d = obj.item()
                if (token_key not in d) and _is_index_dict(d):
                    for k in _sort_keys(d.keys()):
                        tok = _extract_tok(d[k], split)
                        trials.append(np.asarray(tok, dtype=np.int64))
                else:
                    tok = _extract_tok(d, split)
                    trials.append(np.asarray(tok, dtype=np.int64))
            else:
                tok = _extract_tok(obj, split)
                trials.append(np.asarray(tok, dtype=np.int64))
            if vocab is not None:
                for ti, t in enumerate(trials):
                    tmin, tmax = int(t.min()), int(t.max())
                    if tmin < 0 or tmax >= vocab: raise ValueError(f"[{npz_path} split={split}] token out of range at trial#{ti}: min={tmin} max={tmax} vocab={vocab}")
            out[split] = trials
    return out

def _update_unigram_counts(counts: np.ndarray, tokNT: np.ndarray):
    vals, cnts = np.unique(tokNT.reshape(-1), return_counts=True)
    counts[vals] += cnts

def compute_unigram_counts_from_sessions(meta, token_key="token", vocab: int = 128):
    counts = np.zeros(vocab, dtype=np.int64)
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):  _update_unigram_counts(counts, tokNT)
    return np.clip(counts, 1, None)

def compute_ngram_counts_from_sessions(meta, n: int, token_key="token", vocab: int = 128):
    assert n >= 1
    ngram_counts = defaultdict(lambda: np.zeros(vocab, dtype=np.int64))
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):
            N, T = tokNT.shape
            if T <= n: continue
            for i in range(N):
                seq = tokNT[i]
                for t in range(n, T):
                    prev = tuple(int(x) for x in seq[t-n:t])
                    nxt = int(seq[t])
                    ngram_counts[prev][nxt] += 1
    return ngram_counts

def compute_bigram_logP_from_sessions(meta, token_key="token", vocab: int = 128, laplace: float = 0.5):
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):
            N, T = tokNT.shape
            if T < 2: continue
            prev = tokNT[:, :-1].reshape(-1)
            nxt  = tokNT[:, 1:].reshape(-1)
            for p, q in zip(prev, nxt): counts[int(p), int(q)] += 1
    probs = (counts + laplace) / (counts.sum(axis=1, keepdims=True) + laplace * vocab)
    return torch.from_numpy(np.log(probs).astype(np.float32))  # [V,V] logP

class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999, track_buffers: bool = False, trainable_only: bool = False):
        self.decay = float(decay)
        self.track_buffers = bool(track_buffers)
        self.trainable_only = bool(trainable_only)
        m = unwrap_compiled(model)
        self.shadow = {}
        self.backup = {}
        for name, p in m.named_parameters():
            if p is None: continue
            if self.trainable_only and (not p.requires_grad): continue
            self.shadow[name] = p.detach().clone()
        if self.track_buffers:
            self.shadow_buf = {}
            for name, b in m.named_buffers(): self.shadow_buf[name] = b.detach().clone()
        else: self.shadow_buf = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = unwrap_compiled(model)
        for name, p in m.named_parameters():
            if name not in self.shadow: continue
            if self.trainable_only and (not p.requires_grad): continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))
        if self.track_buffers and (self.shadow_buf is not None):
            for name, b in m.named_buffers():
                if name in self.shadow_buf: self.shadow_buf[name].copy_(b.detach())

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        m = unwrap_compiled(model)
        self.backup = {}
        for name, p in m.named_parameters():
            if name not in self.shadow: continue
            if self.trainable_only and (not p.requires_grad): continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])
        if self.track_buffers and (self.shadow_buf is not None):
            self.backup_buf = {}
            for name, b in m.named_buffers():
                if name in self.shadow_buf:
                    self.backup_buf[name] = b.detach().clone()
                    b.data.copy_(self.shadow_buf[name])
        else: self.backup_buf = None

    @torch.no_grad()
    def restore(self, model: nn.Module):
        m = unwrap_compiled(model)
        for name, p in m.named_parameters():
            if name in self.backup: p.data.copy_(self.backup[name])
        if self.track_buffers and (self.backup_buf is not None):
            for name, b in m.named_buffers():
                if name in self.backup_buf: b.data.copy_(self.backup_buf[name])
                
def load_ckpt_state_dict(ckpt_path: str, prefer_ema: bool = True):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and ("model" in raw):
        sd_raw = raw["model"]
        sd_ema = raw.get("model_ema", None)
        saved_cfg = raw.get("cfg", None)
    else:
        sd_raw = raw
        sd_ema = None
        saved_cfg = None
    sd_raw = normalize_state_dict_keys(sd_raw)
    sd_ema = normalize_state_dict_keys(sd_ema) if isinstance(sd_ema, dict) else None
    sd_use = sd_ema if (prefer_ema and sd_ema is not None) else sd_raw
    return raw, sd_use, sd_raw, sd_ema, saved_cfg

def _infer_emb_sizes_from_state_dict(sd: dict):
    def _get(name):
        if name in sd: return sd[name]
        if "module."+name in sd: return sd["module."+name]
        return None
    neu = _get("neu_emb.weight")
    ses = _get("ses_emb.weight")
    ckpt_n_neu = int(neu.shape[0]) if neu is not None else None
    ckpt_n_ses = int(ses.shape[0]) if ses is not None else None
    return ckpt_n_neu, ckpt_n_ses

class EpochTimer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.t0 = None

    def start(self):
        _sync_cuda_if_needed(self.device)
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        _sync_cuda_if_needed(self.device)
        return time.perf_counter() - self.t0
