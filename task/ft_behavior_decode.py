from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ft_behavior_dataset import make_loader
from dataset.session_registry import build_sessions_from_registry, filter_sessions_by_root
from model.behavior_decoder.distributed import ddp_cleanup, ddp_print, ddp_setup, is_main_process, set_seed
from model.behavior_decoder.feature_cache import (
    dump_feature_cache_from_loader,
    make_feature_cache_loader,
    validate_feature_cache_meta_or_warn,
    write_or_update_feature_cache_meta,
)
from model.behavior_decoder.ft_eval import eval_per_session, unwrap_ddp
from model.behavior_decoder.ft_trainer import (
    backbone_encode_fulltokens,
    load_backbone_from_ckpt,
    load_head_ckpt,
    remap_head_B_from_base_to_global,
    strip_module_prefix,
    train_head_phase,
)
from model.behavior_decoder.heads import DecodeHeadLowRankUV
from task.FT_configs import register_ft_configs

register_ft_configs()


def _cache_save_dtype(name: str):
    name = str(name).lower()
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp32':
        return torch.float32
    return torch.float16


def _active_cache_tags(do_train_base: bool, do_finetune_heldout: bool, eval_held_in: bool, eval_held_out: bool, loaders: dict) -> list[str]:
    tags = []
    if do_train_base and 'heldin_train' in loaders:
        tags += ['heldin_train', 'heldin_val', 'heldin_test']
    elif eval_held_in and 'heldin_val' in loaders:
        tags += ['heldin_val', 'heldin_test']

    if do_finetune_heldout and 'heldout_train' in loaders:
        tags += ['heldout_train', 'heldout_val', 'heldout_test']
    elif eval_held_out and 'heldout_val' in loaders:
        tags += ['heldout_val', 'heldout_test']

    # final eval in the original script always runs on heldout if heldout loaders exist.
    if (not do_finetune_heldout) and ('heldout_val' in loaders) and (not eval_held_out):
        tags += ['heldout_val', 'heldout_test']

    out = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@hydra.main(version_base='1.3', config_path='../conf/ft', config_name='ft_behavior_tseng')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    ddp_on, device, rank, world_size = ddp_setup(cfg.runtime, preferred_device=str(cfg.runtime.device))
    set_seed(int(cfg.runtime.seed) + int(rank))

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.runtime.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.runtime.tf32)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision(str(cfg.runtime.matmul_precision))
    except Exception:
        pass

    sessions, meta, cache = build_sessions_from_registry(
        registry_json=str(cfg.data.registry_json),
        token_key=str(cfg.data.token_key),
        beh_key=str(cfg.data.beh_key),
        vocab=int(cfg.data.vocab),
        beh_channels=int(cfg.data.beh_channels),
        splits=('train', 'val', 'test'),
    )
    ddp_print(f"[Index] sessions={meta['n_sessions']} n_neurons_total={meta['n_neurons_total']}")

    heldin_sids = filter_sessions_by_root(sessions, str(cfg.data.heldin_root) if cfg.data.heldin_root else None)
    heldout_sids = filter_sessions_by_root(sessions, str(cfg.data.heldout_root) if cfg.data.heldout_root else None)
    if cfg.data.heldin_root and len(heldin_sids) == 0:
        raise ValueError(f'No held-in sessions found under heldin_root={cfg.data.heldin_root}')
    if cfg.data.heldout_root and len(heldout_sids) == 0:
        raise ValueError(f'No held-out sessions found under heldout_root={cfg.data.heldout_root}')
    if not cfg.data.heldin_root:
        all_sids = list(range(meta['n_sessions']))
        heldin_sids = [sid for sid in all_sids if sid not in set(heldout_sids)]
    if not cfg.data.heldout_root:
        heldout_sids = []

    mode = str(cfg.mode).lower()
    do_train_base = bool(cfg.train_base.enabled)
    do_finetune_heldout = bool(cfg.train_heldout.enabled)
    eval_held_in = bool(cfg.eval.eval_held_in or cfg.eval.eval_held_in_init)
    eval_held_out = bool(cfg.eval.eval_held_out or cfg.eval.eval_held_out_init)

    if mode == 'cache_only':
        do_train_base = False
        do_finetune_heldout = False
        eval_held_in = False
        eval_held_out = False
    elif mode == 'train_base':
        do_train_base = True
        do_finetune_heldout = False
    elif mode == 'finetune_heldout':
        do_train_base = False
        do_finetune_heldout = True
    elif mode == 'eval_only':
        do_train_base = False
        do_finetune_heldout = False
    elif mode == 'full_pipeline':
        pass
    else:
        raise ValueError(f'Unknown FT mode: {cfg.mode}')

    backbone = load_backbone_from_ckpt(
        ckpt_path=str(cfg.backbone.ar_ckpt),
        vocab=int(cfg.data.vocab),
        n_neurons=int(meta['n_neurons_total']),
        n_sessions=int(meta['n_sessions']),
        device=device,
        prefer_ema_weights=bool(cfg.backbone.prefer_ema_weights),
    )
    if bool(cfg.backbone.compile):
        if ddp_on:
            ddp_print('[Backbone] DDP mode detected; skip torch.compile for stability.')
        else:
            backbone = torch.compile(backbone, dynamic=bool(cfg.backbone.compile_dynamic))

    head = DecodeHeadLowRankUV(
        d_model=int(backbone.cfg.d_model),
        n_neurons_total=int(meta['n_neurons_total']),
        beh_channels=int(cfg.data.beh_channels),
        up=int(cfg.data.beh_up),
        rank=int(cfg.head.rank),
        dropout=float(cfg.head.dropout),
        conv_kernel=int(cfg.head.smooth_kernel),
        conv_init_identity=bool(cfg.head.conv_init_identity),
        conv_bias=bool(cfg.head.conv_bias),
    ).to(device)

    if cfg.backbone.init_head_ckpt:
        ck = load_head_ckpt(str(cfg.backbone.init_head_ckpt))
        base_head_state = strip_module_prefix(ck['head'])
        try:
            head.load_state_dict(base_head_state, strict=True)
            ddp_print(f'[HeadInit] Loaded head strictly from {cfg.backbone.init_head_ckpt}')
        except Exception as e:
            if not cfg.backbone.init_head_registry_json:
                raise RuntimeError(
                    'Head init failed due to shape mismatch. '
                    'Provide backbone.init_head_registry_json to remap neuron-dependent params. '
                    f'Error={e}'
                )
            ddp_print(f'[HeadInit] Strict load failed ({e}). Remapping B from base registry -> global registry...')
            new_state = {k: v.clone() for k, v in base_head_state.items() if isinstance(v, torch.Tensor)}
            if 'B' not in new_state:
                raise KeyError('init head state missing B param; cannot remap')
            new_state['B'] = remap_head_B_from_base_to_global(
                new_state,
                str(cfg.backbone.init_head_registry_json),
                str(cfg.data.registry_json),
                token_key=str(cfg.data.token_key),
            )
            if ('smooth.weight' in new_state) or ('smooth.bias' in new_state):
                if getattr(head, 'smooth', None) is None:
                    new_state.pop('smooth.weight', None)
                    new_state.pop('smooth.bias', None)
                else:
                    sw = new_state.get('smooth.weight', None)
                    sb = new_state.get('smooth.bias', None)
                    exp_sw = tuple(head.smooth.weight.shape)
                    if (sw is None) or (tuple(sw.shape) != exp_sw):
                        C = int(cfg.data.beh_channels)
                        up = int(cfg.data.beh_up)
                        O = int(C * up)
                        if (sw is not None) and (sw.ndim == 3) and (sw.shape[0] == C) and (exp_sw[0] == O) and (sw.shape[1] == exp_sw[1]) and (sw.shape[2] == exp_sw[2]):
                            new_state['smooth.weight'] = sw.repeat_interleave(up, dim=0)
                            if (sb is not None) and (sb.ndim == 1) and (sb.shape[0] == C):
                                new_state['smooth.bias'] = sb.repeat_interleave(up, dim=0)
                            else:
                                new_state.pop('smooth.bias', None)
                            ddp_print(f'[HeadInit] Expanded smooth.* from C={C} -> O={O} by up={up}')
                        else:
                            new_state.pop('smooth.weight', None)
                            new_state.pop('smooth.bias', None)
                            ddp_print('[HeadInit] Dropped smooth.* from init ckpt due to shape mismatch')
            missing, unexpected = head.load_state_dict(new_state, strict=False)
            ddp_print(f'[HeadInit] Loaded with remap. missing={len(missing)} unexpected={len(unexpected)}')

    if ddp_on and device.startswith('cuda'):
        local_device = int(device.split(':')[-1])
        head = torch.nn.parallel.DistributedDataParallel(
            head,
            device_ids=[local_device],
            output_device=local_device,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    bs = int(cfg.data.batch_size)
    nw = int(cfg.data.num_workers)
    loaders = {}
    if len(heldin_sids) > 0:
        loaders['heldin_train'] = make_loader(sessions, cache, 'train', heldin_sids, bs, shuffle=True, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size)
        loaders['heldin_val'] = make_loader(sessions, cache, 'val', heldin_sids, bs, shuffle=False, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size, pad_to_world_size=False)
        loaders['heldin_test'] = make_loader(sessions, cache, 'test', heldin_sids, bs, shuffle=False, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size, pad_to_world_size=False)
    if len(heldout_sids) > 0:
        loaders['heldout_train'] = make_loader(sessions, cache, 'train', heldout_sids, bs, shuffle=True, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size)
        loaders['heldout_val'] = make_loader(sessions, cache, 'val', heldout_sids, bs, shuffle=False, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size, pad_to_world_size=False)
        loaders['heldout_test'] = make_loader(sessions, cache, 'test', heldout_sids, bs, shuffle=False, seed=int(cfg.runtime.seed), beh_channels=int(cfg.data.beh_channels), beh_up=int(cfg.data.beh_up), num_workers=nw, ddp=ddp_on, rank=rank, world_size=world_size, pad_to_world_size=False)

    use_feature_cache = bool(cfg.cache.enabled) and (bool(cfg.cache.use) or bool(cfg.cache.build))
    use_bf16 = bool(cfg.runtime.use_bf16)
    active_tags = _active_cache_tags(do_train_base, do_finetune_heldout, eval_held_in, eval_held_out, loaders)

    if use_feature_cache and bool(cfg.cache.build) and len(active_tags) > 0:
        use_amp = str(device).startswith('cuda')
        amp_dtype = torch.bfloat16 if (use_amp and bool(use_bf16) and torch.cuda.is_bf16_supported()) else torch.float16
        save_dtype = _cache_save_dtype(str(cfg.cache.dtype))
        for tag in active_tags:
            loader = loaders.get(tag, None)
            if loader is None:
                continue
            dump_feature_cache_from_loader(
                tag=tag,
                backbone=backbone,
                encode_fn=backbone_encode_fulltokens,
                dl=loader,
                cache_dir=str(cfg.cache.dir),
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                force=bool(cfg.cache.force),
                save_dtype=save_dtype,
                max_batches=int(cfg.cache.max_batches),
            )
        write_or_update_feature_cache_meta(str(cfg.cache.dir), active_tags, cfg, save_dtype, device=device, inferred_only=False)

    use_cached_features = False
    if use_feature_cache and bool(cfg.cache.use) and len(active_tags) > 0:
        validate_feature_cache_meta_or_warn(
            str(cfg.cache.dir),
            active_tags,
            cfg,
            _cache_save_dtype(str(cfg.cache.dtype)),
            allow_mismatch=bool(cfg.cache.allow_mismatch),
        )
        use_cached_features = True
        if 'heldin_train' in active_tags:
            loaders['heldin_train'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldin_train', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=True, allow_empty=False)
        if 'heldin_val' in active_tags:
            loaders['heldin_val'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldin_val', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=False, allow_empty=bool(cfg.cache.allow_empty_eval))
        if 'heldin_test' in active_tags:
            loaders['heldin_test'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldin_test', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=False, allow_empty=bool(cfg.cache.allow_empty_eval))
        if 'heldout_train' in active_tags:
            loaders['heldout_train'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldout_train', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=True, allow_empty=False)
        if 'heldout_val' in active_tags:
            loaders['heldout_val'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldout_val', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=False, allow_empty=bool(cfg.cache.allow_empty_eval))
        if 'heldout_test' in active_tags:
            loaders['heldout_test'] = make_feature_cache_loader(str(cfg.cache.dir), 'heldout_test', num_workers=nw, seed=int(cfg.runtime.seed), shuffle=False, allow_empty=bool(cfg.cache.allow_empty_eval))
        try:
            cache.clear()
        except Exception:
            pass

    save_dir = os.path.join(str(cfg.paths.save_dir), str(cfg.run_name))
    os.makedirs(save_dir, exist_ok=True)

    if mode != 'eval_only' and eval_held_in and 'heldin_val' in loaders:
        if is_main_process():
            print('\n=== EVAL (INIT) HELD-IN ===')
        eval_per_session(backbone, head, loaders['heldin_val'], beh_up=int(cfg.data.beh_up), device=device, tag='INIT/HELDIN/VAL', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
        eval_per_session(backbone, head, loaders['heldin_test'], beh_up=int(cfg.data.beh_up), device=device, tag='INIT/HELDIN/TEST', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
    if mode != 'eval_only' and eval_held_out and 'heldout_val' in loaders:
        if is_main_process():
            print('\n=== EVAL (INIT) HELD-OUT ===')
        eval_per_session(backbone, head, loaders['heldout_val'], beh_up=int(cfg.data.beh_up), device=device, tag='INIT/HELDOUT/VAL', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
        eval_per_session(backbone, head, loaders['heldout_test'], beh_up=int(cfg.data.beh_up), device=device, tag='INIT/HELDOUT/TEST', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)

    if mode == 'cache_only':
        ddp_print('[Mode] cache_only done.')
        if ddp_on:
            ddp_cleanup()
        return

    if do_train_base and 'heldin_train' in loaders:
        ddp_print('\n=== TRAIN HEAD on HELD-IN ===')
        head = train_head_phase(
            phase_name='BASE',
            backbone=backbone,
            head=head,
            dl_tr=loaders['heldin_train'],
            dl_va=loaders['heldin_val'],
            dl_te=loaders['heldin_test'],
            device=device,
            lr=float(cfg.train_base.lr),
            weight_decay=float(cfg.train_base.weight_decay),
            epochs=int(cfg.train_base.epochs),
            warmup_frac=float(cfg.train_base.warmup_frac),
            ft_mode='all',
            freeze_neuron_ids=None,
            save_dir=save_dir,
            eval_every_epochs=int(cfg.train_base.eval_every_epochs),
            do_test_during_train=bool(cfg.train_base.do_test_during_train),
            max_eval_batches=int(cfg.eval.max_eval_batches),
            print_each_session=bool(cfg.eval.print_each_session),
            print_limit_sessions=int(cfg.eval.print_limit_sessions),
            use_cached_features=use_cached_features,
            use_bf16=use_bf16,
        )

    if do_finetune_heldout and ('heldout_train' in loaders):
        ddp_print('\n=== FINETUNE HEAD on HELD-OUT ===')
        freeze_ids = None
        if str(cfg.train_heldout.ft_mode) == 'newrows':
            heldin_neuron_ids = []
            for s in sessions:
                sid = int(s['session_id'])
                if sid in set(heldin_sids):
                    off = int(s['neuron_offset'])
                    n = int(s['n_neurons'])
                    heldin_neuron_ids.extend(list(range(off, off + n)))
            freeze_ids = torch.tensor(heldin_neuron_ids, dtype=torch.long)
            ddp_print(f'[FT] newrows mode: freeze {len(heldin_neuron_ids)} held-in neuron rows in head.B')
        head = train_head_phase(
            phase_name='HELDOUT',
            backbone=backbone,
            head=head,
            dl_tr=loaders['heldout_train'],
            dl_va=loaders['heldout_val'],
            dl_te=loaders['heldout_test'],
            device=device,
            lr=float(cfg.train_heldout.lr),
            weight_decay=float(cfg.train_heldout.weight_decay),
            epochs=int(cfg.train_heldout.epochs),
            warmup_frac=float(cfg.train_heldout.warmup_frac),
            ft_mode=str(cfg.train_heldout.ft_mode),
            freeze_neuron_ids=freeze_ids,
            save_dir=save_dir,
            eval_every_epochs=int(cfg.train_heldout.eval_every_epochs),
            do_test_during_train=bool(cfg.train_heldout.do_test_during_train),
            max_eval_batches=int(cfg.eval.max_eval_batches),
            print_each_session=bool(cfg.eval.print_each_session),
            print_limit_sessions=int(cfg.eval.print_limit_sessions),
            use_cached_features=use_cached_features,
            use_bf16=use_bf16,
        )

    if mode in ('eval_only', 'full_pipeline', 'train_base', 'finetune_heldout'):
        final_eval_held_in = (eval_held_in if mode == 'eval_only' else do_train_base)
        final_eval_held_out = (eval_held_out if mode == 'eval_only' else ('heldout_val' in loaders))
        if final_eval_held_in and 'heldin_val' in loaders:
            if is_main_process():
                print('\n=== FINAL EVAL HELD-IN ===')
            eval_per_session(backbone, head, loaders['heldin_val'], beh_up=int(cfg.data.beh_up), device=device, tag='FINAL/HELDIN/VAL', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
            eval_per_session(backbone, head, loaders['heldin_test'], beh_up=int(cfg.data.beh_up), device=device, tag='FINAL/HELDIN/TEST', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
        if final_eval_held_out and 'heldout_val' in loaders:
            if is_main_process():
                print('\n=== FINAL EVAL HELD-OUT ===')
            eval_per_session(backbone, head, loaders['heldout_val'], beh_up=int(cfg.data.beh_up), device=device, tag='FINAL/HELDOUT/VAL', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)
            eval_per_session(backbone, head, loaders['heldout_test'], beh_up=int(cfg.data.beh_up), device=device, tag='FINAL/HELDOUT/TEST', print_each_session=bool(cfg.eval.print_each_session), max_batches=int(cfg.eval.max_eval_batches), print_limit_sessions=int(cfg.eval.print_limit_sessions), use_cached_features=use_cached_features, encode_fn=backbone_encode_fulltokens, use_bf16=use_bf16)

    if is_main_process():
        torch.save({'head': unwrap_ddp(head).state_dict(), 'meta': meta, 'cfg': OmegaConf.to_container(cfg, resolve=True)}, os.path.join(save_dir, 'final_head.pt'))
        print(f'\n[Done] Saved final_head.pt to {save_dir}')
    if ddp_on:
        ddp_cleanup()


if __name__ == '__main__':
    main()
