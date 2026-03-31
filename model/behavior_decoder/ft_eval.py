from __future__ import annotations

import time

import numpy as np
import torch

from .distributed import ddp_is_initialized, ddp_max_float, ddp_print, get_rank, is_main_process


def safe_mean(xs):
    if len(xs) == 0:
        return float('nan')
    return float(np.mean(xs))


def unwrap_ddp(m):
    return m.module if hasattr(m, 'module') else m


@torch.no_grad()
def eval_per_session(
    backbone,
    head,
    dl,
    beh_up: int,
    device: str,
    tag: str,
    print_each_session: bool = True,
    max_batches: int = -1,
    print_limit_sessions: int = -1,
    var_min: float = 1e-6,
    min_len: int = 3,
    use_cached_features: bool = False,
    encode_fn=None,
    use_bf16: bool = True,
):
    t0 = time.time()
    ddp_print(f'[RANK {get_rank()}] start evaluation: {tag}', flush=True)

    backbone.eval()
    head_mod = unwrap_ddp(head)
    head.eval()

    S = int(getattr(getattr(backbone, 'cfg', None), 'n_sessions', 0))
    if S <= 0:
        raise ValueError('eval_per_session: backbone.cfg.n_sessions must be > 0')
    C = int(getattr(head_mod, 'C', 1))
    eps = 1e-8

    frame_sse_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    frame_cnt_S = torch.zeros((S,), device=device, dtype=torch.float64)
    trial_mse_sum_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    trial_cnt_S = torch.zeros((S,), device=device, dtype=torch.float64)
    corr_sum_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    corr_cnt_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    r2_sum_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    r2_cnt_SC = torch.zeros((S, C), device=device, dtype=torch.float64)
    batch_cnt_S = torch.zeros((S,), device=device, dtype=torch.float64)

    use_amp = str(device).startswith('cuda')
    amp_dtype = torch.bfloat16 if (use_amp and bool(use_bf16) and torch.cuda.is_bf16_supported()) else torch.float16
    n_batches_seen = 0
    with torch.inference_mode():
        for batch in dl:
            n_batches_seen += 1
            if int(max_batches) > 0 and n_batches_seen > int(max_batches):
                break

            use_cached = use_cached_features and ('hBNTD' in batch)
            if use_cached:
                h = batch['hBNTD'].to(device, non_blocking=True)
                behBCT = batch['behBCT']
                beh_lengths = batch['beh_lengths']
                session_idB = batch['session_idB']
                neuron_idsBN = batch['neuron_idsBN'].to(device, non_blocking=True)
            else:
                xBNT = batch['xBNT']
                lengths = batch['lengths']
                behBCT = batch['behBCT']
                beh_lengths = batch['beh_lengths']
                session_idB = batch['session_idB']
                neuron_offset = int(batch['neuron_offset'][0].item())
                h, neuron_idsBN = encode_fn(backbone, xBNT, lengths, session_idB, neuron_offset, device)

            with torch.autocast(device_type=('cuda' if use_amp else 'cpu'), dtype=amp_dtype, enabled=use_amp):
                pred = head(h, neuron_idsBN)
            pred = pred.float()
            tgt = behBCT.to(device, non_blocking=True).float()
            Tuse = min(int(pred.shape[-1]), int(tgt.shape[-1]))
            pred = pred[:, :, :Tuse]
            tgt = tgt[:, :, :Tuse]
            B = pred.shape[0]
            mask_beh = (torch.arange(Tuse, device=device).unsqueeze(0) < beh_lengths.to(device, non_blocking=True).reshape(B, 1))
            sid = int(session_idB[0].item())
            if sid < 0 or sid >= S:
                raise ValueError(f'session_id out of range: sid={sid} S={S}')

            se_bct = (pred - tgt) ** 2
            mask_f = mask_beh.to(pred.dtype)
            sse_c = (se_bct * mask_f[:, None, :]).sum(dim=(0, 2))
            nfrm = mask_f.sum().double()
            frame_sse_SC[sid] += sse_c.double()
            frame_cnt_S[sid] += nfrm

            denom_trial = mask_f.sum(dim=1).clamp_min(1.0)
            sse_bc = (se_bct * mask_f[:, None, :]).sum(dim=2)
            mse_bc = (sse_bc / denom_trial[:, None]).double()
            trial_mse_sum_SC[sid] += mse_bc.sum(dim=0)
            trial_cnt_S[sid] += float(B)

            for b in range(B):
                mb = mask_beh[b]
                L = int(mb.sum().item())
                if L < int(min_len):
                    continue
                for c in range(C):
                    y0 = tgt[b, c, mb]
                    x0 = pred[b, c, mb]
                    if float(y0.var(unbiased=False).item()) < float(var_min):
                        continue
                    if float(x0.var(unbiased=False).item()) >= float(var_min):
                        xc = x0 - x0.mean()
                        yc = y0 - y0.mean()
                        denom = (xc.std(unbiased=False) * yc.std(unbiased=False)).clamp_min(eps)
                        corr = (xc * yc).mean() / denom
                        corr_sum_SC[sid, c] += corr.double()
                        corr_cnt_SC[sid, c] += 1.0
                    ss_res = ((y0 - x0) ** 2).sum()
                    ss_tot = ((y0 - y0.mean()) ** 2).sum()
                    if float(ss_tot.item()) < float(eps):
                        continue
                    r2 = 1.0 - ss_res / ss_tot
                    r2_sum_SC[sid, c] += r2.double()
                    r2_cnt_SC[sid, c] += 1.0
            batch_cnt_S[sid] += 1.0

    if ddp_is_initialized():
        import torch.distributed as dist
        for t in (frame_sse_SC, frame_cnt_S, trial_mse_sum_SC, trial_cnt_S, corr_sum_SC, corr_cnt_SC, r2_sum_SC, r2_cnt_SC, batch_cnt_S):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    elapsed_local = time.time() - t0
    elapsed = ddp_max_float(elapsed_local, device=device)
    if not is_main_process():
        return {'micro': {}, 'macro': {}, 'per_session': [], 'elapsed_s': float(elapsed)}

    valid = (trial_cnt_S > 0) & (frame_cnt_S > 0)
    n_valid_sessions = int(valid.sum().item())
    mse_frame_mean_SC = frame_sse_SC / frame_cnt_S.clamp_min(1.0)[:, None]
    mse_trial_mean_SC = trial_mse_sum_SC / trial_cnt_S.clamp_min(1.0)[:, None]
    corr_mean_SC = torch.where(corr_cnt_SC > 0, corr_sum_SC / corr_cnt_SC.clamp_min(1.0), torch.full_like(corr_sum_SC, float('nan')))
    r2_mean_SC = torch.where(r2_cnt_SC > 0, r2_sum_SC / r2_cnt_SC.clamp_min(1.0), torch.full_like(r2_sum_SC, float('nan')))

    total_frames = frame_cnt_S.sum().clamp_min(1.0)
    micro_mse_frame_c = (frame_sse_SC.sum(dim=0) / total_frames).tolist()
    micro_mse_frame = float(np.mean(micro_mse_frame_c)) if len(micro_mse_frame_c) else float('nan')
    total_trials = trial_cnt_S.sum().clamp_min(1.0)
    micro_mse_trial_c = (trial_mse_sum_SC.sum(dim=0) / total_trials).tolist()
    micro_mse_trial = float(np.mean(micro_mse_trial_c)) if len(micro_mse_trial_c) else float('nan')
    micro_corr = (corr_sum_SC.sum(dim=0) / corr_cnt_SC.sum(dim=0).clamp_min(1.0)).tolist()
    micro_r2 = (r2_sum_SC.sum(dim=0) / r2_cnt_SC.sum(dim=0).clamp_min(1.0)).tolist()

    if n_valid_sessions > 0:
        macro_mse_frame_c = mse_frame_mean_SC[valid].mean(dim=0).tolist()
        macro_mse_frame = float(np.mean(macro_mse_frame_c))
        macro_mse_trial_c = mse_trial_mean_SC[valid].mean(dim=0).tolist()
        macro_mse_trial = float(np.mean(macro_mse_trial_c))
        macro_corr = torch.nanmean(corr_mean_SC[valid], dim=0).tolist()
        macro_r2 = torch.nanmean(r2_mean_SC[valid], dim=0).tolist()
    else:
        macro_mse_frame = float('nan')
        macro_mse_trial = float('nan')
        macro_mse_frame_c = [float('nan')] * C
        macro_mse_trial_c = [float('nan')] * C
        macro_corr = [float('nan')] * C
        macro_r2 = [float('nan')] * C

    ddp_print(
        f'[{tag}] time={elapsed:.2f}s '
        f'MICRO mse_frame={micro_mse_frame:.6g} mse_frame_c={["%.6g" % x for x in micro_mse_frame_c]} '
        f'mse_trial={micro_mse_trial:.6g} mse_trial_c={["%.6g" % x for x in micro_mse_trial_c]} '
        f'corr={["%.4f" % x for x in micro_corr]} r2={["%.4f" % x for x in micro_r2]} '
        f'valid_corr_cnt={corr_cnt_SC.sum(dim=0).tolist()} valid_r2_cnt={r2_cnt_SC.sum(dim=0).tolist()}'
    )
    ddp_print(
        f'[{tag}] time={elapsed:.2f}s '
        f'MACRO mse_frame={macro_mse_frame:.6g} mse_frame_c={["%.6g" % x for x in macro_mse_frame_c]} '
        f'mse_trial={macro_mse_trial:.6g} mse_trial_c={["%.6g" % x for x in macro_mse_trial_c]} '
        f'corr={["%.4f" % x for x in macro_corr]} r2={["%.4f" % x for x in macro_r2]}'
    )

    session_rows = []
    if n_valid_sessions > 0:
        sids = torch.where(valid)[0].tolist()
        if int(print_limit_sessions) >= 0:
            sids = sids[:int(print_limit_sessions)]
        for sid in sids:
            mf_c = mse_frame_mean_SC[sid].tolist()
            mt_c = mse_trial_mean_SC[sid].tolist()
            cc = corr_mean_SC[sid].tolist()
            rr = r2_mean_SC[sid].tolist()
            nb = int(batch_cnt_S[sid].item())
            nt = int(trial_cnt_S[sid].item())
            session_rows.append((sid, mf_c, mt_c, cc, rr, nb, nt))
            if print_each_session:
                ddp_print(
                    f'  - sid={sid:03d} n_trials={nt:4d} n_batches={nb:4d} '
                    f'mse_frame={float(np.mean(mf_c)):.6g} mse_frame_c={["%.6g" % x for x in mf_c]} '
                    f'mse_trial={float(np.mean(mt_c)):.6g} mse_trial_c={["%.6g" % x for x in mt_c]} '
                    f'corr={["%.4f" % x for x in cc]} r2={["%.4f" % x for x in rr]}'
                )

    return {
        'micro': {'mse': micro_mse_frame, 'mse_c': micro_mse_frame_c, 'mse_trial': micro_mse_trial, 'mse_trial_c': micro_mse_trial_c, 'corr': micro_corr, 'r2': micro_r2},
        'macro': {'mse': macro_mse_frame, 'mse_c': macro_mse_frame_c, 'mse_trial': macro_mse_trial, 'mse_trial_c': macro_mse_trial_c, 'corr': macro_corr, 'r2': macro_r2},
        'per_session': session_rows,
        'elapsed_s': float(elapsed),
    }
