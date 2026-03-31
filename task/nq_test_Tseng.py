from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from util import set_seed
from model import VQForCalcium
from task.nq_vq_configs_test_Tseng import register_tokenize_configs, register_configs
register_configs()
register_tokenize_configs()

def strip_orig_mod_prefix(state_dict: dict):
    for pfx in ("_orig_mod.", "orig_mod."):
        if any(k.startswith(pfx) for k in state_dict.keys()): return {k.replace(pfx, "", 1): v for k, v in state_dict.items()}
    return state_dict

def align_to_mod_trace(x_nt: np.ndarray, mod: int) -> Tuple[np.ndarray, int]:
    r = x_nt.shape[1] % mod
    if r == 0: return x_nt, 0
    need = mod - r
    last = x_nt[:, -1:]
    return np.concatenate([x_nt, np.repeat(last, repeats=need, axis=1)], axis=1), r

def align_to_mod_lastdim(arr: np.ndarray, mod: int, r: int) -> np.ndarray:
    if r == 0: return arr
    need = mod - r
    last = arr[..., -1:]
    return np.concatenate([arr, np.repeat(last, repeats=need, axis=-1)], axis=-1)

def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if np.std(a) < eps or np.std(b) < eps: return float("nan")
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.sqrt((a0*a0).sum()) * np.sqrt((b0*b0).sum())) + eps
    return float((a0*b0).sum() / denom)

REQUIRED_KEYS = {"train", "val", "test", "meta"}

def is_valid_result_npz(path: str) -> bool:
    if (not os.path.exists(path)) or os.path.getsize(path) == 0: return False
    try:
        with np.load(path, allow_pickle=True) as d:
            keys = set(d.files)
            if not REQUIRED_KEYS.issubset(keys): return False
            _ = d["meta"].item() if hasattr(d["meta"], "item") else d["meta"]
        return True
    except Exception: return False

def atomic_savez_compressed(final_path: str, **kwargs):
    tmp_path = final_path + ".tmp"
    if os.path.exists(tmp_path):
        try: os.remove(tmp_path)
        except Exception: pass
    with open(tmp_path, "wb") as f: np.savez_compressed(f, **kwargs)
    os.replace(tmp_path, final_path)

@torch.no_grad()
def export_split(
    model: nn.Module,
    total_data: Dict[str, Any],
    split: str,
    device: str,
    n_emb: int,
    neuron_chunk: int = 512,
    use_amp: bool = False,
    align_mod: int = 4,
):
    X = total_data[f"{split}_padded_X"]       # [B, N, T_max]
    L = total_data[f"{split}_lengths"]        # [B]
    trial_ids = total_data[f"{split}_trial_ids"]
    Y = total_data.get(f"{split}_padded_Y", None)
    out: Dict[str, Any] = {}
    token_id_list = []
    corr_list = []
    r2_list = []
    for idx, trial_raw in enumerate(X):
        trial_len = int(L[idx])
        if trial_len <= 1: continue
        trial_raw = np.asarray(trial_raw[:, :trial_len], dtype=np.float32)  # [N, T_valid]
        trial_aligned, r = align_to_mod_trace(trial_raw, align_mod)         # [N, T_aligned]
        N, T = trial_aligned.shape
        gt_np = trial_aligned
        decoded_parts = []
        token_parts = []
        for s in range(0, N, neuron_chunk):
            e = min(s + neuron_chunk, N)
            chunk = trial_aligned[s:e]  # [Nc, T] CPU
            trial_tensor = torch.from_numpy(chunk).float().unsqueeze(1).to(device, non_blocking=True)  # [Nc,1,T]
            if use_amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): decoded, _, _, _, token_ids = model(trial_tensor, if_training=False)
            else: decoded, _, _, _, token_ids = model(trial_tensor, if_training=False)
            decoded_np = decoded.squeeze(1).detach().cpu().numpy()    # [Nc, T]
            token_np = token_ids.detach().cpu().numpy()              # [Nc, P] or [Nc,...]
            decoded_parts.append(decoded_np)
            token_parts.append(token_np)
            del trial_tensor, decoded, token_ids
        decoded_np = np.concatenate(decoded_parts, axis=0)  # [N, T]
        token_np   = np.concatenate(token_parts, axis=0)
        token_id_list.extend(token_np.reshape(-1).tolist())
        corrs = []
        for n in range(N):
            c = safe_corr(gt_np[n], decoded_np[n])
            if not np.isnan(c): corrs.append(c)
        corr_mean = float(np.mean(corrs)) if len(corrs) else float("nan")
        corr_list.append(corr_mean)
        r2s = []
        for n in range(N):
            y = gt_np[n]
            yhat = decoded_np[n]
            denom = np.sum((y - y.mean())**2)
            if denom < 1e-12: continue
            r2s.append(1.0 - (np.sum((y - yhat)**2) / denom))
        r2 = float(np.mean(r2s)) if len(r2s) else float("nan")
        r2_list.append(r2)
        beh = None
        if Y is not None:
            beh_raw = Y[idx, :, :trial_len]
            beh = align_to_mod_lastdim(beh_raw, align_mod, r)
        out[str(idx)] = {
            "token": token_np,
            "trace": gt_np,
            "trial_id": trial_ids[idx],
            "behavior": beh,
            "remainder": int(r),
            "corr": corr_mean,
            "r2": r2,
        }
        if idx % 20 == 0: print(f"[{split}] idx={idx:4d}  N={N:4d}  T={T:5d}  corr={corr_mean:.4f}  r2={r2:.4f}")
        if device.startswith("cuda") and (idx % 100 == 0): torch.cuda.empty_cache()
    usage = np.zeros(n_emb, dtype=np.int64)
    if len(token_id_list):
        vals, counts = np.unique(np.asarray(token_id_list, dtype=np.int64), return_counts=True)
        usage[vals] = counts
    total = int(usage.sum()) if int(usage.sum()) > 0 else 1
    p = usage / max(int(usage.sum()), 1)
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log2(p_nz))) if p_nz.size else 0.0
    topk = np.sort(usage)[::-1]
    stats = {
        "num_trials": int(len(out)),
        "mean_corr": float(np.mean(corr_list)) if len(corr_list) else float("nan"),
        "mean_r2": float(np.mean(r2_list)) if len(r2_list) else float("nan"),
        "used_codes": int((usage > 0).sum()),
        "top1_pct": float(topk[0] / total) if topk.size else 0.0,
        "top10_pct": float(topk[:10].sum() / total) if topk.size else 0.0,
        "entropy_bits": entropy,
        "perplexity": float(2 ** entropy),
        "token_count": int(usage.sum()),
    }
    return out, stats

@hydra.main(version_base="1.3", config_path="../conf/nq", config_name="vq_test_Tseng")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.runtime.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.runtime.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.runtime.tf32)
    try: torch.set_float32_matmul_precision(str(cfg.runtime.matmul_precision))
    except Exception: pass
    device = str(cfg.runtime.device)
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = os.path.join(str(cfg.data.data_root), str(cfg.tokenize.sub))
    pat = str(cfg.tokenize.glob_pattern).format(keywords=str(cfg.tokenize.keywords))
    files = glob.glob(os.path.join(root_path, pat), recursive=True)
    files = [f for f in files if f.endswith(".npz")]
    files.sort()
    print(f"[Files] root={root_path}  matched={len(files)}")
    out_root_path = os.path.join(str(cfg.tokenize.out_root_base), str(cfg.tokenize.sub))
    os.makedirs(out_root_path, exist_ok=True)
    m = cfg.model
    use_gumbel = bool(m.use_gumbel)
    use_gumbel_hard = bool(m.use_gumbel_hard)
    if bool(cfg.tokenize.force_gumbel_off):
        use_gumbel = False
        use_gumbel_hard = False
    model = VQForCalcium(
        int(m.discretization_window),
        int(m.overlap_window),
        int(m.n_emb),
        int(m.dim_emb),
        int(m.heads),
        int(m.trans_layer_num_enc),
        int(m.trans_layer_num_dec),
        dropout_ratio=float(m.dropout_ratio),
        decay=float(m.decay),
        epsilon=float(m.epsilon),
        use_gumbel=use_gumbel,
        use_gumbel_hard=use_gumbel_hard,
        temperature=float(m.temperature_high),
        reset_threshold_ratio=float(m.reset_threshold_ratio),
        dead_code_ema_reset_val=float(m.dead_code_ema_reset_val),
        use_periodic_kmeans_recluster=bool(m.use_periodic_kmeans_recluster),
        z_e_buffer_capacity=int(m.z_e_buffer_capacity),
        min_z_e_for_recluster=int(m.min_z_e_for_recluster),
        temperature_entropy=float(m.temperature_high),
        max_ar_k=int(m.max_ar_k),
        lookahead_tokens=int(m.lookahead_tokens),
        encoder_causal=bool(m.encoder_causal),
        decoder_causal=bool(m.decoder_causal),
    ).to(device).eval()
    ckpt_path = str(cfg.tokenize.ckpt_path)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = strip_orig_mod_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if len(missing) or len(unexpected):
        print("[Warn] strict load mismatch")
        print(" missing:", missing[:10])
        print(" unexpected:", unexpected[:10])
    if bool(cfg.tokenize.compile):
        model = torch.compile(model, dynamic=bool(cfg.tokenize.compile_dynamic), mode=str(cfg.tokenize.compile_mode))
        print("[Compile] enabled")
    align_mod = int(cfg.tokenize.align_mod) if cfg.tokenize.align_mod is not None else int(m.discretization_window)
    for data_path in files:
        filename = os.path.basename(data_path)
        stem = filename[:-4] if filename.endswith(".npz") else Path(filename).stem
        out_path = os.path.join(out_root_path, f"{cfg.tokenize.date}_e{int(cfg.tokenize.epoch)}_{stem}.npz")
        if bool(cfg.tokenize.resume_skip_valid) and is_valid_result_npz(out_path):
            print(f"[Resume] Skip done session: {filename}")
            continue
        if os.path.exists(out_path) and bool(cfg.tokenize.remove_invalid):
            if not is_valid_result_npz(out_path):
                print(f"[Resume] Remove invalid and rerun: {out_path}")
                try: os.remove(out_path)
                except Exception as e: print(f"[Resume] Remove failed: {e}")
        print(f"[Run] {filename}")
        print(f"[Out] {out_path}")
        try:
            with np.load(data_path, allow_pickle=True) as d: total_data = dict(d)
            train_dict, train_stats = export_split(
                model, total_data, "train", device=device, n_emb=int(m.n_emb),
                neuron_chunk=int(cfg.tokenize.neuron_chunk),
                use_amp=bool(cfg.tokenize.use_amp),
                align_mod=align_mod,
            )
            val_dict, val_stats = export_split(
                model, total_data, "val", device=device, n_emb=int(m.n_emb),
                neuron_chunk=int(cfg.tokenize.neuron_chunk),
                use_amp=bool(cfg.tokenize.use_amp),
                align_mod=align_mod,
            )
            test_dict, test_stats = export_split(
                model, total_data, "test", device=device, n_emb=int(m.n_emb),
                neuron_chunk=int(cfg.tokenize.neuron_chunk),
                use_amp=bool(cfg.tokenize.use_amp),
                align_mod=align_mod,
            )
            meta = {
                "data_path": data_path,
                "ckpt_path": ckpt_path,
                "sub": str(cfg.tokenize.sub),
                "date": str(cfg.tokenize.date),
                "epoch": int(cfg.tokenize.epoch),
                "lookahead_tokens": int(m.lookahead_tokens),
                "discretization_window": int(m.discretization_window),
                "overlap_window": int(m.overlap_window),
                "n_emb": int(m.n_emb),
                "dim_emb": int(m.dim_emb),
                "stats": {"train": train_stats, "val": val_stats, "test": test_stats},
            }
            atomic_savez_compressed(
                out_path,
                train=np.array([train_dict], dtype=object),
                val=np.array([val_dict], dtype=object),
                test=np.array([test_dict], dtype=object),
                meta=np.array([meta], dtype=object),
            )
            print("Saved ONE file:", out_path)
            print("train:", train_stats)
            print("val:", val_stats)
            print("test:", test_stats)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] Skip session: {filename}")
                if device.startswith("cuda"): torch.cuda.empty_cache()
                continue
            raise

if __name__ == "__main__":
    main()