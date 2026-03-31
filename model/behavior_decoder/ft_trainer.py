from __future__ import annotations

import math
import os
import re
import time
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset.session_registry import _to_posix, build_offsets_from_registry
from model.dynamics_transformer.dt_layers import AxialARCFG
from model.dynamics_transformer.dual_axis_transformer import AxialAR
from model.dynamics_transformer.dt_utility import _infer_emb_sizes_from_state_dict, load_ckpt_state_dict
from .distributed import ddp_is_initialized, ddp_max_float, ddp_print, is_main_process
from .ft_eval import eval_per_session, unwrap_ddp


def strip_module_prefix(state: dict) -> dict:
    if not isinstance(state, dict) or len(state) == 0:
        return state
    if any(str(k).startswith('module.') for k in state.keys()):
        return {str(k)[len('module.'):]: v for k, v in state.items()}
    return state


def _infer_cfg_from_backbone_state(state: Dict[str, torch.Tensor], vocab_default: int, n_neurons: int, n_sessions: int) -> Dict[str, Any]:
    tok_emb = state.get('tok_emb.weight', None)
    d_model = int(tok_emb.shape[1]) if tok_emb is not None else 512
    n_layers = 0
    for k in state.keys():
        m = re.match(r'blocks\.(\d+)\.', k)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
    if n_layers <= 0:
        n_layers = 6
    return dict(
        vocab=int(vocab_default),
        n_neurons=int(n_neurons),
        n_sessions=int(n_sessions),
        d_model=int(d_model),
        n_heads=8,
        n_layers=int(n_layers),
        d_ff=2048,
        dropout=0.1,
        emb_dropout=0.1,
        attn_dropout=0.1,
        use_abs_time_emb=False,
    )


def _build_cfg_from_kwargs(kwargs: Dict[str, Any]) -> AxialARCFG:
    if is_dataclass(AxialARCFG):
        allowed = {f.name for f in fields(AxialARCFG)}
    else:
        allowed = set(kwargs.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return AxialARCFG(**filtered)


def load_backbone_from_ckpt(ckpt_path: str, vocab: int, n_neurons: int, n_sessions: int, device: str, prefer_ema_weights: bool = True):
    _, sd_use, _, _, saved_cfg = load_ckpt_state_dict(ckpt_path, prefer_ema=bool(prefer_ema_weights))
    state = strip_module_prefix(sd_use)
    ckpt_n_neu, ckpt_n_ses = _infer_emb_sizes_from_state_dict(state)
    if ckpt_n_neu is not None and int(ckpt_n_neu) < int(n_neurons):
        raise ValueError(f'Backbone ckpt has fewer neuron embeddings than current registry: ckpt={ckpt_n_neu}, registry={n_neurons}')
    if ckpt_n_ses is not None and int(ckpt_n_ses) < int(n_sessions):
        raise ValueError(f'Backbone ckpt has fewer session embeddings than current registry: ckpt={ckpt_n_ses}, registry={n_sessions}')

    # Follow the original FT single-file semantics more closely:
    # start from state-dict inference using CURRENT registry totals, then only
    # borrow optional non-size hyperparameters from saved_cfg when available.
    cfg_kwargs = _infer_cfg_from_backbone_state(state, vocab_default=vocab, n_neurons=n_neurons, n_sessions=n_sessions)
    if isinstance(saved_cfg, dict):
        for k in ('d_model', 'n_heads', 'n_layers', 'd_ff', 'dropout', 'emb_dropout', 'attn_dropout', 'use_abs_time_emb', 'rope_theta'):
            if k in saved_cfg:
                cfg_kwargs[k] = saved_cfg[k]
        # Critically, do NOT trust saved n_neurons / n_sessions sizes here; the
        # original script tied these to the current global registry.
        cfg_kwargs['vocab'] = int(saved_cfg.get('vocab', vocab))
        cfg_kwargs['n_neurons'] = int(n_neurons)
        cfg_kwargs['n_sessions'] = int(n_sessions)
    cfg = _build_cfg_from_kwargs(cfg_kwargs)

    model = AxialAR(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        ddp_print('[Backbone] unexpected keys:', unexpected[:20])
    if len(missing) > 0:
        ddp_print('[Backbone] missing keys:', missing[:20])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_head_ckpt(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'head' in ckpt:
        return ckpt
    return {'head': ckpt}


def remap_head_B_from_base_to_global(head_state_base: Dict[str, torch.Tensor], base_registry_json: str, global_registry_json: str, token_key: str) -> torch.Tensor:
    B_base = head_state_base['B']
    O, N_base_total, R = B_base.shape
    base_paths, base_offsets, base_ns = build_offsets_from_registry(base_registry_json, token_key)
    glob_paths, glob_offsets, glob_ns = build_offsets_from_registry(global_registry_json, token_key)
    base_map = {_to_posix(p): (i, base_offsets[i], base_ns[i]) for i, p in enumerate(base_paths)}
    glob_map = {_to_posix(p): (i, glob_offsets[i], glob_ns[i]) for i, p in enumerate(glob_paths)}
    N_global_total = int(sum(glob_ns))
    B_global = torch.randn((O, N_global_total, R), dtype=B_base.dtype) * 0.02
    copied = 0
    for p_posix, (_, off_b, nb) in base_map.items():
        if p_posix not in glob_map:
            continue
        _, off_g, ng = glob_map[p_posix]
        if nb != ng:
            raise ValueError(f'Neuron count mismatch for shared session: {p_posix} baseN={nb} globalN={ng}')
        src = slice(off_b, off_b + nb)
        dst = slice(off_g, off_g + nb)
        B_global[:, dst, :] = B_base[:, src, :]
        copied += nb
    ddp_print(f'[HeadRemap] Copied neurons from base->global for {copied}/{N_base_total} base neurons. Global total={N_global_total}.')
    return B_global


@torch.inference_mode()
def backbone_encode_fulltokens(backbone: AxialAR, xBNT: torch.Tensor, lengths: torch.Tensor, session_idB: torch.Tensor, neuron_offset: int, device: str):
    xBNT = xBNT.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    session_idB = session_idB.to(device, non_blocking=True)
    B, N, Tm = xBNT.shape
    time_posBT = torch.arange(Tm, device=device).unsqueeze(0).expand(B, Tm)
    time_kpmaskBT = (torch.arange(Tm, device=device).unsqueeze(0) < lengths.reshape(B, 1))
    neuron_idsBN = (torch.arange(N, device=device).unsqueeze(0).expand(B, N) + int(neuron_offset)).long()
    neuron_idsBNT = neuron_idsBN[:, :, None].expand(B, N, Tm)
    h = backbone.encode(xBNT, neuron_idsBNT, time_posBT, time_kpmaskBT, session_idB=session_idB)
    return h, neuron_idsBN


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_head_phase(
    phase_name: str,
    backbone: AxialAR,
    head: nn.Module,
    dl_tr,
    dl_va,
    dl_te,
    device: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_frac: float,
    ft_mode: str = 'all',
    freeze_neuron_ids: Optional[torch.Tensor] = None,
    save_dir: Optional[str] = None,
    eval_every_epochs: int = 1,
    do_test_during_train: bool = False,
    max_eval_batches: int = -1,
    print_each_session: bool = False,
    print_limit_sessions: int = -1,
    use_cached_features: bool = False,
    use_bf16: bool = True,
):
    use_amp = device.startswith('cuda')
    amp_dtype = torch.bfloat16 if (use_amp and bool(use_bf16) and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    assert ft_mode in ('all', 'newrows')
    head_mod = unwrap_ddp(head)
    if not ddp_is_initialized():
        head_mod.to(device)
    head.train()

    for p in head_mod.parameters():
        p.requires_grad_(False)
    if ft_mode == 'all':
        for p in head_mod.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([p for p in head_mod.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    else:
        head_mod.B.requires_grad_(True)
        opt = torch.optim.AdamW([{'params': [head_mod.B], 'weight_decay': 0.0}], lr=float(lr), weight_decay=0.0)

    trainable = [p for p in head_mod.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError('No trainable parameters in head (check ft_mode).')

    total_steps = int(len(dl_tr) * max(1, int(epochs)))
    warmup_steps = int(total_steps * float(warmup_frac))
    sched = cosine_with_warmup(opt, warmup_steps=warmup_steps, total_steps=total_steps)
    best_val = float('inf')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    freeze_idx = freeze_neuron_ids.to(device) if (freeze_neuron_ids is not None) else None

    for ep in range(1, int(epochs) + 1):
        head.train()
        if hasattr(getattr(dl_tr, 'batch_sampler', None), 'set_epoch'):
            dl_tr.batch_sampler.set_epoch(ep)

        loss_sum = torch.tensor(0.0, device=device)
        loss_cnt = torch.tensor(0.0, device=device)
        t0 = time.time()
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            use_cached = use_cached_features and ('hBNTD' in batch)
            if use_cached:
                h = batch['hBNTD'].to(device, non_blocking=True)
                neuron_idsBN = batch['neuron_idsBN'].to(device, non_blocking=True)
                behBCT = batch['behBCT']
                beh_lengths = batch['beh_lengths']
            else:
                xBNT = batch['xBNT']
                lengths = batch['lengths']
                behBCT = batch['behBCT']
                beh_lengths = batch['beh_lengths']
                session_idB = batch['session_idB']
                neuron_offset = int(batch['neuron_offset'][0].item())
                h, neuron_idsBN = backbone_encode_fulltokens(backbone, xBNT, lengths, session_idB, neuron_offset, device)

            with torch.autocast(device_type=('cuda' if use_amp else 'cpu'), dtype=amp_dtype, enabled=use_amp):
                pred = head(h, neuron_idsBN)
                tgt = behBCT.to(device, non_blocking=True).float()
                Tuse = min(int(pred.shape[-1]), int(tgt.shape[-1]))
                pred = pred[:, :, :Tuse]
                tgt = tgt[:, :, :Tuse]
                B = pred.shape[0]
                mask_beh = (torch.arange(Tuse, device=device).unsqueeze(0) < beh_lengths.to(device, non_blocking=True).reshape(B, 1))
                mask_f = mask_beh.to(pred.dtype)[:, None, :]
                denom = mask_f.sum().clamp_min(1e-8)
                loss = (((pred - tgt) ** 2) * mask_f).sum() / denom

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if ft_mode == 'newrows' and (freeze_idx is not None) and (head_mod.B.grad is not None):
                    head_mod.B.grad.index_fill_(1, freeze_idx, 0.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if ft_mode == 'newrows' and (freeze_idx is not None) and (head_mod.B.grad is not None):
                    head_mod.B.grad.index_fill_(1, freeze_idx, 0.0)
                opt.step()
            sched.step()
            loss_sum += loss.detach()
            loss_cnt += 1.0

        if ddp_is_initialized():
            import torch.distributed as dist
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cnt, op=dist.ReduceOp.SUM)
        train_loss = float((loss_sum / loss_cnt.clamp_min(1.0)).item())
        train_time = ddp_max_float(time.time() - t0, device=device)
        lr_now = opt.param_groups[0]['lr']

        val_mse = float('inf')
        val_time = 0.0
        test_time = 0.0
        do_eval = (int(eval_every_epochs) > 0 and ep > 1 and (((ep - 1) % int(eval_every_epochs) == 0) or ep == int(epochs)))
        if do_eval:
            val_stats = eval_per_session(
                backbone, head, dl_va, beh_up=unwrap_ddp(head).up, device=device,
                tag=f'{phase_name}/VAL/EP{ep:03d}',
                print_each_session=bool(print_each_session),
                max_batches=int(max_eval_batches),
                print_limit_sessions=int(print_limit_sessions),
                use_cached_features=use_cached_features,
                encode_fn=backbone_encode_fulltokens,
                use_bf16=use_bf16,
            )
            val_time = float(val_stats.get('elapsed_s', 0.0))
            if is_main_process():
                val_mse = float(val_stats['micro']['mse'])
            if bool(do_test_during_train):
                te_stats = eval_per_session(
                    backbone, head, dl_te, beh_up=unwrap_ddp(head).up, device=device,
                    tag=f'{phase_name}/TEST/EP{ep:03d}',
                    print_each_session=bool(print_each_session),
                    max_batches=int(max_eval_batches),
                    print_limit_sessions=int(print_limit_sessions),
                    use_cached_features=use_cached_features,
                    encode_fn=backbone_encode_fulltokens,
                    use_bf16=use_bf16,
                )
                test_time = float(te_stats.get('elapsed_s', 0.0))
            ddp_print(
                f'[{phase_name}] EP{ep:03d} lr={lr_now:.3e} '
                f'train_mse={train_loss:.6g} train_time={train_time:.2f}s '
                f'val_mse={val_mse:.6g} val_time={val_time:.2f}s '
                f'test_time={test_time:.2f}s'
            )
        else:
            ddp_print(f'[{phase_name}] EP{ep:03d} lr={lr_now:.3e} train_mse={train_loss:.6g} train_time={train_time:.2f}s')

        if save_dir and is_main_process() and do_eval:
            torch.save({'head': head_mod.state_dict(), 'epoch': ep, 'val_mse': val_mse}, os.path.join(save_dir, f'last_{phase_name.lower()}.pt'))
            if val_mse < best_val:
                best_val = val_mse
                torch.save({'head': head_mod.state_dict(), 'epoch': ep, 'val_mse': val_mse}, os.path.join(save_dir, f'best_{phase_name.lower()}.pt'))

    return head
