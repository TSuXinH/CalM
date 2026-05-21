from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import random
import copy
from omegaconf import DictConfig, OmegaConf

from dataset.dt_dataset import build_multisession_loaders_fixedN_v2
from model import (
    AxialAR,
    AxialARCFG,
    build_vq_decoder,
    compute_bigram_logP_from_sessions,
    compute_ngram_counts_from_sessions,
    compute_unigram_counts_from_sessions,
    eval_multisession_val_vq_recon_rollout,
    load_ckpt_state_dict,
    run_axial_ar,
)
from model.dynamics_transformer.dt_utility import _infer_emb_sizes_from_state_dict
from task.DT_configs import register_dt_configs
from util.distributed import (
    _dist_avail_and_initialized,
    ddp_prepare_registry,
    ddp_setup,
    get_local_rank,
    get_rank,
    get_canonical_heldout_registry,
    is_main_process,
    log_print,
    make_distributed_dataloader,
    prepare_canonical_heldout_registry,
)

register_dt_configs()


def _auto_device(dev: str) -> str:
    dev = str(dev)
    if dev == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _neighbor_builder_from_vq(vq_model, vocab: int, neighbor_k: int, device: str):
    with torch.no_grad():
        E = vq_model.codebook(torch.arange(vocab, device=device).reshape(1, -1)).squeeze(0)
        E = F.normalize(E, dim=-1)
        sim = E @ E.t()
        sim.fill_diagonal_(-1.0)
        k = int(max(1, neighbor_k))
        return sim.topk(k=k, dim=-1).indices.detach().cpu().numpy()


def _build_loaders(cfg: DictConfig, *, heldout: bool = False, registry_override: Optional[str] = None):
    data_root = str(cfg.data.heldout_root) if heldout else str(cfg.data.data_root)
    assert data_root, 'data root is required'
    if registry_override is not None:
        registry_json = registry_override
    else:
        registry_json = str(cfg.data.registry_json_heldout) if heldout and cfg.data.registry_json_heldout else (str(cfg.data.registry_json) if cfg.data.registry_json else None)
    return build_multisession_loaders_fixedN_v2(
        data_root=data_root,
        n_sub=int(cfg.data.n_sub),
        pattern=str(cfg.data.pattern),
        token_key=str(cfg.data.token_key),
        vocab=int(cfg.data.vocab),
        batch_size=int(cfg.data.batch_size),
        eval_batch_size=int(cfg.data.eval_batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        preload=bool(cfg.data.preload),
        exclude_dir_prefix=tuple(cfg.data.exclude_dir_prefix),
        registry_json=registry_json,
        base_seed=int(cfg.runtime.seed),
        train_split=str(cfg.data.train_split),
        val_split=str(cfg.data.val_split),
        test_split=str(cfg.data.test_split),
    )


def _wrap_loaders_for_ddp(cfg: DictConfig, dl_tr, dl_va, dl_te, *, shuffle_train: bool):
    if not _dist_avail_and_initialized():
        return dl_tr, dl_va, dl_te

    dl_tr_full, dl_va_full, dl_te_full = dl_tr, dl_va, dl_te
    dl_tr = make_distributed_dataloader(
        dl_tr,
        shuffle=bool(shuffle_train),
        seed=int(cfg.runtime.seed),
        pad=bool(cfg.runtime.sampler_pad),
    )
    dl_va = make_distributed_dataloader(
        dl_va,
        shuffle=False,
        seed=int(cfg.runtime.seed),
        pad=bool(cfg.runtime.sampler_pad),
    )
    dl_te = make_distributed_dataloader(
        dl_te,
        shuffle=False,
        seed=int(cfg.runtime.seed),
        pad=bool(cfg.runtime.sampler_pad),
    )
    setattr(dl_tr, '_full', dl_tr_full)
    setattr(dl_va, '_full', dl_va_full)
    setattr(dl_te, '_full', dl_te_full)
    return dl_tr, dl_va, dl_te


def _load_stats_inputs(cfg: DictConfig, meta: dict, device: str):
    unigram_counts = compute_unigram_counts_from_sessions(
        meta,
        token_key=str(cfg.data.token_key),
        vocab=int(cfg.data.vocab),
    ) if bool(cfg.train.use_cf_weights) else None

    bigram_logP = compute_bigram_logP_from_sessions(
        meta,
        token_key=str(cfg.data.token_key),
        vocab=int(cfg.data.vocab),
        laplace=float(cfg.train.ngram_laplace),
    ) if bool(cfg.train.use_bigram_rescore) else None

    if bool(cfg.train.use_ngram_kd):
        ngram_counts = compute_ngram_counts_from_sessions(
            meta,
            n=int(cfg.train.ngram_order),
            token_key=str(cfg.data.token_key),
            vocab=int(cfg.data.vocab),
        )
        run_axial_ar._ngram_counts = ngram_counts  # type: ignore[attr-defined]

    neighbor_idx = None
    if bool(cfg.train.use_tok_neighbor_noise) and str(cfg.vq.vq_state).strip():
        neighbor_idx = _neighbor_builder_from_vq

    if isinstance(bigram_logP, torch.Tensor):
        bigram_logP = bigram_logP.to(device)

    return unigram_counts, bigram_logP, neighbor_idx


@hydra.main(version_base='1.3', config_path='../conf/dt', config_name='train_dt_Tseng')
def main(cfg: DictConfig) -> None:
    requested_device = 'cuda' if str(cfg.runtime.device) == 'auto' else str(cfg.runtime.device)
    ddp_on, ddp_device = ddp_setup(requested_device, backend=str(cfg.runtime.ddp_backend))

    if ddp_on:
        device = ddp_device
    else:
        device = _auto_device(str(cfg.runtime.device))

    print(OmegaConf.to_yaml(cfg))

    seed = int(cfg.runtime.seed) + (get_rank() if ddp_on else 0)
    _seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(cfg.runtime.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.runtime.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.runtime.tf32)
    try:
        torch.set_float32_matmul_precision(str(cfg.runtime.matmul_precision))
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(bool(cfg.runtime.deterministic))
    except Exception:
        pass

    mode = str(cfg.train.mode).lower()
    if mode not in ('train', 'eval_test', 'finetune_heldout'):
        raise ValueError(f'Unknown train.mode={cfg.train.mode}')

    if mode == 'eval_test':
        # Always build base meta first, just like the original ABO script.
        registry_json_use, commit_registry = ddp_prepare_registry(
            str(cfg.data.registry_json) if cfg.data.registry_json else None,
            per_rank=bool(cfg.runtime.registry_per_rank),
        )
        dl_tr, dl_va, dl_te, meta = _build_loaders(cfg, heldout=False, registry_override=registry_json_use)
        commit_registry()
        dl_tr, dl_va, dl_te = _wrap_loaders_for_ddp(cfg, dl_tr, dl_va, dl_te, shuffle_train=False)

        assert cfg.train.eval_ckpt, 'train.eval_ckpt is required for eval_test'
        assert str(cfg.vq.vq_state).strip(), 'eval_test requires a non-empty vq.vq_state'

        raw, sd_use, _sd_raw, _sd_ema, saved_cfg = load_ckpt_state_dict(
            str(cfg.train.eval_ckpt),
            prefer_ema=bool(cfg.train.eval_use_ema_weights),
        )
        ckpt_cfg = saved_cfg if isinstance(saved_cfg, dict) else {}

        def _g(key: str, default: Any) -> Any:
            return ckpt_cfg.get(key, default)

        n_neu_ckpt, n_ses_ckpt = _infer_emb_sizes_from_state_dict(sd_use)
        if n_neu_ckpt is None:
            n_neu_ckpt = int(meta['n_neurons_total'])
        if n_ses_ckpt is None:
            n_ses_ckpt = int(meta['n_sessions'])

        m = cfg.model
        ar_cfg = AxialARCFG(
            vocab=int(_g('vocab', int(cfg.data.vocab))),
            n_neurons=int(_g('n_neurons', int(n_neu_ckpt))),
            n_sessions=int(_g('n_sessions', int(n_ses_ckpt))),
            d_model=int(_g('d_model', int(m.d_model))),
            n_layers=int(_g('n_layers', int(m.n_layers))),
            n_heads=int(_g('n_heads', int(m.n_heads))),
            d_ff=int(_g('d_ff', int(m.d_ff))),
            dropout=float(_g('dropout', float(m.dropout))),
            emb_dropout=float(_g('emb_dropout', float(m.emb_dropout))),
            attn_dropout=float(_g('attn_dropout', float(m.attn_dropout))),
            train_window_T=int(_g('train_window_T', int(m.train_window_T))),
            eval_use_full_trial=bool(_g('eval_use_full_trial', bool(m.eval_use_full_trial))),
            use_abs_time_emb=bool(_g('use_abs_time_emb', bool(m.use_abs_time_emb))),
        )
        model_raw = AxialAR(ar_cfg).to(device)
        missing, unexpected = model_raw.load_state_dict(sd_use, strict=False)
        print(f'[Eval] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')
        if len(missing) < 20:
            print('  missing:', missing)
        if len(unexpected) < 20:
            print('  unexpected:', unexpected)

        model = model_raw
        if bool(cfg.train.compile):
            model = torch.compile(model_raw, dynamic=bool(cfg.train.compile_dynamic))
            print('[Eval] torch.compile enabled')
        if _dist_avail_and_initialized():
            lrk = get_local_rank()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[lrk],
                output_device=lrk,
                broadcast_buffers=False,
            )

        vq_model, vq_stride = build_vq_decoder(
            vq_state_path=str(cfg.vq.vq_state),
            discretization_window=int(cfg.vq.vq_disc_win),
            overlap_window=int(cfg.vq.vq_overlap),
            n_emb=int(cfg.vq.vq_n_emb),
            dim_emb=int(cfg.vq.vq_dim_emb),
            heads=int(cfg.vq.vq_heads),
            enc_layers=int(cfg.vq.vq_enc_layers),
            dec_layers=int(cfg.vq.vq_dec_layers),
            encoder_causal=bool(cfg.vq.vq_encoder_causal),
            decoder_causal=bool(cfg.vq.vq_decoder_causal),
            device=device,
            lookahead_tokens=int(cfg.vq.lookahead_tokens),
        )

        eval_target = str(cfg.train.eval_target).lower()
        if eval_target not in ('base', 'heldout', 'both'):
            raise ValueError(f'Unknown train.eval_target={cfg.train.eval_target}')

        off_s = int(raw.get('id_offset_session', 0) or 0) if isinstance(raw, dict) else 0
        off_n = int(raw.get('id_offset_neuron', 0) or 0) if isinstance(raw, dict) else 0
        if (off_s == 0 and off_n == 0) and (eval_target in ('heldout', 'both')):
            off_s = int(meta.get('n_sessions', 0))
            off_n = int(meta.get('n_neurons_total', 0))

        if eval_target in ('base', 'both'):
            eval_multisession_val_vq_recon_rollout(
                model=model,
                ema=None,
                dl_va=dl_te,
                meta=meta,
                vq_model=vq_model,
                vq_stride=vq_stride,
                prefix_tokens=int(cfg.train.gen_prefix),
                tag='TEST_BASE',
                device=device,
                use_ema=False,
                id_offset_session=0,
                id_offset_neuron=0,
            )

        if eval_target in ('heldout', 'both'):
            assert cfg.data.heldout_root, 'data.heldout_root is required when eval_target is heldout/both'
            heldout_registry = get_canonical_heldout_registry(
                (str(cfg.data.registry_json_heldout) if cfg.data.registry_json_heldout else None),
                str(cfg.paths.ckpt_dir),
                (str(cfg.data.registry_json) if cfg.data.registry_json else None),
            )
            prepare_canonical_heldout_registry((str(cfg.data.registry_json) if cfg.data.registry_json else None), heldout_registry)
            registry_json_use, commit_registry = ddp_prepare_registry(
                heldout_registry,
                per_rank=bool(cfg.runtime.registry_per_rank),
            )
            dl_tr_ho, dl_va_ho, dl_te_ho, meta_ho = _build_loaders(cfg, heldout=True, registry_override=registry_json_use)
            commit_registry()
            dl_tr_ho, dl_va_ho, dl_te_ho = _wrap_loaders_for_ddp(cfg, dl_tr_ho, dl_va_ho, dl_te_ho, shuffle_train=(not bool(cfg.runtime.deterministic)))

            meta_ho_shift = dict(meta_ho)
            shifted = []
            for s in meta_ho.get('sessions', []):
                s2 = dict(s)
                s2['session_id'] = int(s2['session_id']) + off_s
                s2['neuron_offset'] = int(s2['neuron_offset']) + off_n
                shifted.append(s2)
            meta_ho_shift['sessions'] = shifted
            meta_ho_shift['n_sessions'] = int(off_s + int(meta_ho.get('n_sessions', 0)))
            meta_ho_shift['n_neurons_total'] = int(off_n + int(meta_ho.get('n_neurons_total', 0)))

            eval_multisession_val_vq_recon_rollout(
                model=model,
                ema=None,
                dl_va=dl_te_ho,
                meta=meta_ho_shift,
                vq_model=vq_model,
                vq_stride=vq_stride,
                prefix_tokens=int(cfg.train.gen_prefix),
                tag='TEST_HELDOUT',
                device=device,
                use_ema=False,
                id_offset_session=off_s,
                id_offset_neuron=off_n,
            )
        return

    if mode == 'finetune_heldout':
        assert cfg.data.heldout_root, 'data.heldout_root required for finetune_heldout'
        ckpt_init = cfg.train.heldout_init_ckpt or cfg.train.init_ckpt
        assert ckpt_init, 'Need train.heldout_init_ckpt or train.init_ckpt'

        heldout_registry = get_canonical_heldout_registry(
            (str(cfg.data.registry_json_heldout) if cfg.data.registry_json_heldout else None),
            str(cfg.paths.ckpt_dir),
            (str(cfg.data.registry_json) if cfg.data.registry_json else None),
        )
        prepare_canonical_heldout_registry((str(cfg.data.registry_json) if cfg.data.registry_json else None), heldout_registry)
        registry_json_use, commit_registry = ddp_prepare_registry(
            heldout_registry,
            per_rank=bool(cfg.runtime.registry_per_rank),
        )
        dl_tr_ho, dl_va_ho, dl_te_ho, meta_ho = _build_loaders(cfg, heldout=True, registry_override=registry_json_use)
        commit_registry()
        dl_tr_ho, dl_va_ho, dl_te_ho = _wrap_loaders_for_ddp(cfg, dl_tr_ho, dl_va_ho, dl_te_ho, shuffle_train=(not bool(cfg.runtime.deterministic)))

        _raw, _sd_use, sd_raw, _sd_ema, _saved_cfg = load_ckpt_state_dict(str(ckpt_init), prefer_ema=False)
        ckpt_n_neu = int(sd_raw['neu_emb.weight'].shape[0])
        ckpt_n_ses = int(sd_raw['ses_emb.weight'].shape[0])

        meta_ft = copy.deepcopy(meta_ho)
        meta_ft['n_sessions'] = ckpt_n_ses + meta_ho['n_sessions']
        meta_ft['n_neurons_total'] = ckpt_n_neu + meta_ho['n_neurons_total']
        meta_ft['sessions'] = []
        for s in meta_ho['sessions']:
            s2 = dict(s)
            s2['session_id'] = int(s2['session_id']) + ckpt_n_ses
            s2['neuron_offset'] = int(s2['neuron_offset']) + ckpt_n_neu
            meta_ft['sessions'].append(s2)

        if is_main_process():
            print(
                f"[Data] sessions={meta_ft['n_sessions']}  total_neurons={meta_ft['n_neurons_total']}  "
                f"train_batches={len(dl_tr_ho)}  val_batches={len(dl_va_ho)}"
            )

        unigram_counts, bigram_logP, neighbor_idx = _load_stats_inputs(cfg, meta_ft, device)
        cfg_overrides = dict(OmegaConf.to_container(cfg.model, resolve=True))
        cfg_overrides['train_mode'] = str(cfg.train.heldout_train_mode)
        ft_lr = float(cfg.train.heldout_lr) if cfg.train.heldout_lr is not None else float(cfg.optim.lr)

        run_axial_ar(
            train_trials=None,
            val_trials=None,
            vocab=int(cfg.data.vocab),
            device=device,
            dl_tr=dl_tr_ho,
            dl_va=dl_va_ho,
            meta=meta_ft,
            init_ckpt=str(ckpt_init),
            cfg_overrides=cfg_overrides,
            epochs=int(cfg.train.heldout_epochs),
            lr=float(ft_lr),
            min_lr=float(cfg.optim.min_lr),
            warmup_ratio=float(cfg.optim.warmup_ratio),
            lr_schedule=str(cfg.optim.lr_schedule),
            weight_decay=float(cfg.optim.weight_decay),
            gen_every=int(cfg.train.gen_every),
            gen_prefix=int(cfg.train.gen_prefix),
            gen_outdir=str(cfg.paths.gen_outdir),
            ckpt_every=int(cfg.train.ckpt_every),
            ckpt_dir=str(cfg.paths.ckpt_dir),
            use_ema=bool(cfg.train.use_ema),
            ema_decay=float(cfg.train.ema_decay),
            ema_eval_only=bool(cfg.train.ema_eval_only),
            save_ema_weights=bool(cfg.train.save_ema_weights),
            use_mtp=bool(cfg.train.use_mtp),
            mtp_horizons=int(cfg.train.mtp_horizons),
            mtp_hidden=int(cfg.train.mtp_hidden),
            mtp_weight=str(cfg.train.mtp_weight),
            mtp_geom_gamma=float(cfg.train.mtp_geom_gamma),
            mtp_warmup_epochs=int(cfg.train.mtp_warmup_epochs),
            mtp_lambda=float(cfg.train.mtp_lambda),
            use_scheduled_sampling=bool(cfg.train.use_scheduled_sampling),
            ss_prob=float(cfg.train.ss_prob),
            ss_block_len=int(cfg.train.ss_block_len),
            cf_counts=unigram_counts,
            use_ngram_kd=bool(cfg.train.use_ngram_kd),
            ngram_order=int(cfg.train.ngram_order),
            ngram_alpha=float(cfg.train.ngram_alpha),
            ngram_laplace=float(cfg.train.ngram_laplace),
            use_bigram_rescore=bool(cfg.train.use_bigram_rescore),
            bigram_logP=bigram_logP,
            bigram_alpha=float(cfg.train.bigram_alpha),
            neighbor_idx=neighbor_idx,
            tok_noise_p=float(cfg.train.tok_noise_p),
            neighbor_k=int(cfg.train.neighbor_k),
            decoding_beam=int(cfg.train.decoding_beam),
            beam_lambda_emb_smooth=float(cfg.train.beam_lambda_emb_smooth),
            post_smooth_win=int(cfg.train.post_smooth_win),
            beam_expand_neurons=int(cfg.train.beam_expand_neurons),
            beam_per_neuron_topk=int(cfg.train.beam_per_neuron_topk),
            decode_temp=float(cfg.train.decode_temp),
            eval_detail_every=int(cfg.train.eval_detail_every),
            vq_state=str(cfg.vq.vq_state),
            vq_disc_win=int(cfg.vq.vq_disc_win),
            vq_overlap=int(cfg.vq.vq_overlap),
            vq_n_emb=int(cfg.vq.vq_n_emb),
            vq_dim_emb=int(cfg.vq.vq_dim_emb),
            vq_heads=int(cfg.vq.vq_heads),
            vq_enc_layers=int(cfg.vq.vq_enc_layers),
            vq_dec_layers=int(cfg.vq.vq_dec_layers),
            vq_encoder_causal=bool(cfg.vq.vq_encoder_causal),
            vq_decoder_causal=bool(cfg.vq.vq_decoder_causal),
            lookahead_tokens=int(cfg.vq.lookahead_tokens),
            compile_model=bool(cfg.train.compile),
            compile_dynamic=bool(cfg.train.compile_dynamic),
            init_use_ema_weights=bool(cfg.train.heldout_init_use_ema_weights),
            id_offset_neuron=int(ckpt_n_neu),
            id_offset_session=int(ckpt_n_ses),
            use_bf16=bool(cfg.runtime.use_bf16),
        )
        return

    # mode == train
    registry_json_use, commit_registry = ddp_prepare_registry(
        str(cfg.data.registry_json) if cfg.data.registry_json else None,
        per_rank=bool(cfg.runtime.registry_per_rank),
    )
    dl_tr, dl_va, dl_te, meta = _build_loaders(cfg, heldout=False, registry_override=registry_json_use)
    commit_registry()
    dl_tr, dl_va, dl_te = _wrap_loaders_for_ddp(cfg, dl_tr, dl_va, dl_te, shuffle_train=(not bool(cfg.runtime.deterministic)))

    if is_main_process():
        print(
            f"[Data] sessions={meta['n_sessions']}  total_neurons={meta['n_neurons_total']}  "
            f"train_batches={len(dl_tr)}  val_batches={len(dl_va)}"
        )

    unigram_counts, bigram_logP, neighbor_idx = _load_stats_inputs(cfg, meta, device)
    cfg_overrides = dict(OmegaConf.to_container(cfg.model, resolve=True))
    cfg_overrides['train_mode'] = str(cfg.train.train_mode)

    run_axial_ar(
        train_trials=None,
        val_trials=None,
        vocab=int(cfg.data.vocab),
        device=device,
        dl_tr=dl_tr,
        dl_va=dl_va,
        meta=meta,
        cfg_overrides=cfg_overrides,
        epochs=int(cfg.train.epochs),
        lr=float(cfg.optim.lr),
        min_lr=float(cfg.optim.min_lr),
        warmup_ratio=float(cfg.optim.warmup_ratio),
        lr_schedule=str(cfg.optim.lr_schedule),
        weight_decay=float(cfg.optim.weight_decay),
        init_ckpt=(str(cfg.train.init_ckpt) if cfg.train.init_ckpt else None),
        gen_every=int(cfg.train.gen_every),
        gen_prefix=int(cfg.train.gen_prefix),
        gen_outdir=str(cfg.paths.gen_outdir),
        ckpt_every=int(cfg.train.ckpt_every),
        ckpt_dir=str(cfg.paths.ckpt_dir),
        use_ema=bool(cfg.train.use_ema),
        ema_decay=float(cfg.train.ema_decay),
        ema_eval_only=bool(cfg.train.ema_eval_only),
        save_ema_weights=bool(cfg.train.save_ema_weights),
        use_mtp=bool(cfg.train.use_mtp),
        mtp_horizons=int(cfg.train.mtp_horizons),
        mtp_hidden=int(cfg.train.mtp_hidden),
        mtp_weight=str(cfg.train.mtp_weight),
        mtp_geom_gamma=float(cfg.train.mtp_geom_gamma),
        mtp_warmup_epochs=int(cfg.train.mtp_warmup_epochs),
        mtp_lambda=float(cfg.train.mtp_lambda),
        use_scheduled_sampling=bool(cfg.train.use_scheduled_sampling),
        ss_prob=float(cfg.train.ss_prob),
        ss_block_len=int(cfg.train.ss_block_len),
        cf_counts=unigram_counts,
        use_ngram_kd=bool(cfg.train.use_ngram_kd),
        ngram_order=int(cfg.train.ngram_order),
        ngram_alpha=float(cfg.train.ngram_alpha),
        ngram_laplace=float(cfg.train.ngram_laplace),
        use_bigram_rescore=bool(cfg.train.use_bigram_rescore),
        bigram_logP=bigram_logP,
        bigram_alpha=float(cfg.train.bigram_alpha),
        neighbor_idx=neighbor_idx,
        tok_noise_p=float(cfg.train.tok_noise_p),
        neighbor_k=int(cfg.train.neighbor_k),
        decoding_beam=int(cfg.train.decoding_beam),
        beam_lambda_emb_smooth=float(cfg.train.beam_lambda_emb_smooth),
        post_smooth_win=int(cfg.train.post_smooth_win),
        beam_expand_neurons=int(cfg.train.beam_expand_neurons),
        beam_per_neuron_topk=int(cfg.train.beam_per_neuron_topk),
        decode_temp=float(cfg.train.decode_temp),
        eval_detail_every=int(cfg.train.eval_detail_every),
        vq_state=str(cfg.vq.vq_state),
        vq_disc_win=int(cfg.vq.vq_disc_win),
        vq_overlap=int(cfg.vq.vq_overlap),
        vq_n_emb=int(cfg.vq.vq_n_emb),
        vq_dim_emb=int(cfg.vq.vq_dim_emb),
        vq_heads=int(cfg.vq.vq_heads),
        vq_enc_layers=int(cfg.vq.vq_enc_layers),
        vq_dec_layers=int(cfg.vq.vq_dec_layers),
        vq_encoder_causal=bool(cfg.vq.vq_encoder_causal),
        vq_decoder_causal=bool(cfg.vq.vq_decoder_causal),
        lookahead_tokens=int(cfg.vq.lookahead_tokens),
        compile_model=bool(cfg.train.compile),
        compile_dynamic=bool(cfg.train.compile_dynamic),
        use_bf16=bool(cfg.runtime.use_bf16),
    )


if __name__ == '__main__':
    main()
