from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from .distributed import ddp_is_initialized, ddp_max_float, ddp_print, get_rank, get_world_size, is_main_process
from dataset.session_registry import _fingerprint_file


def _safe_makedirs(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _json_dump_atomic(obj: dict, path: str):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def _feature_cache_meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'feature_cache_meta.json')


def _feature_cache_done_flag(cache_dir: str, tag: str, rank: int) -> str:
    return os.path.join(cache_dir, tag, f'DONE_rank{rank}.txt')


def feature_cache_is_done(cache_dir: str, tag: str, rank: int) -> bool:
    return os.path.exists(_feature_cache_done_flag(cache_dir, tag, rank))


def mark_feature_cache_done(cache_dir: str, tag: str, rank: int):
    _safe_makedirs(os.path.join(cache_dir, tag))
    with open(_feature_cache_done_flag(cache_dir, tag, rank), 'w', encoding='utf-8') as f:
        f.write('ok\n')


def list_feature_cache_files(cache_dir: str, tag: str, rank: int) -> List[str]:
    d = os.path.join(cache_dir, tag)
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf'^{re.escape(tag)}_rank{rank}_step\d{{6}}\.pt$')
    files = []
    for fn in os.listdir(d):
        if pat.match(fn):
            files.append(os.path.join(d, fn))
    files.sort()
    return files


def _collect_cache_stats_local(cache_dir: str, tag: str, rank: int) -> dict:
    files = list_feature_cache_files(cache_dir, tag, rank)
    total_bytes = 0
    for fp in files:
        try:
            total_bytes += int(os.path.getsize(fp))
        except Exception:
            pass
    return {
        'rank': int(rank),
        'tag': str(tag),
        'done_flag': bool(feature_cache_is_done(cache_dir, tag, rank)),
        'n_files': int(len(files)),
        'total_bytes': int(total_bytes),
    }


def _cfg_data(cfg, key: str, default=None):
    return getattr(getattr(cfg, 'data', None), key, default)


def _cfg_backbone(cfg, key: str, default=None):
    return getattr(getattr(cfg, 'backbone', None), key, default)


def _cfg_runtime(cfg, key: str, default=None):
    return getattr(getattr(cfg, 'runtime', None), key, default)


def write_or_update_feature_cache_meta(
    cache_dir: str,
    tags: List[str],
    cfg: Any,
    save_dtype: torch.dtype,
    device: str,
    inferred_only: bool = False,
):
    _safe_makedirs(cache_dir)
    rank = get_rank()
    world_size = get_world_size()
    local = {
        'rank': int(rank),
        'world_size': int(world_size),
        'per_tag': {t: _collect_cache_stats_local(cache_dir, t, rank) for t in tags},
    }
    if ddp_is_initialized():
        gathered = [None for _ in range(world_size)]
        import torch.distributed as dist
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]

    if is_main_process():
        tag_stats = {}
        for t in tags:
            per_rank = []
            for g in gathered:
                if g is None:
                    continue
                if 'per_tag' in g and t in g['per_tag']:
                    per_rank.append(g['per_tag'][t])
            per_rank = sorted(per_rank, key=lambda x: x.get('rank', 0))
            tag_stats[t] = {
                'per_rank': per_rank,
                'all_ranks_done': all(bool(r.get('done_flag')) for r in per_rank) if len(per_rank) else False,
                'total_files': int(sum(int(r.get('n_files', 0)) for r in per_rank)),
                'total_bytes': int(sum(int(r.get('total_bytes', 0)) for r in per_rank)),
            }

        meta_path = _feature_cache_meta_path(cache_dir)
        prev = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    prev = json.load(f)
            except Exception:
                prev = None

        now = time.time()
        meta = {
            'schema_version': 1,
            'created_at_unix': float(now),
            'created_at_local': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
            'inferred_only': bool(inferred_only),
            'runtime': {'world_size': int(world_size)},
            'config': {
                'save_dtype': str(save_dtype).replace('torch.', ''),
                'seed': int(_cfg_runtime(cfg, 'seed', 0)),
                'batch_size': int(_cfg_data(cfg, 'batch_size', 0)),
                'beh_channels': int(_cfg_data(cfg, 'beh_channels', 0)),
                'beh_up': int(_cfg_data(cfg, 'beh_up', 0)),
                'token_key': str(_cfg_data(cfg, 'token_key', '')),
                'beh_key': str(_cfg_data(cfg, 'beh_key', '')),
            },
            'inputs': {
                'registry_json': _fingerprint_file(_cfg_data(cfg, 'registry_json', '')),
                'ar_ckpt': _fingerprint_file(_cfg_backbone(cfg, 'ar_ckpt', '')),
                'heldin_root': str(_cfg_data(cfg, 'heldin_root', '') or '').replace('\\', '/'),
                'heldout_root': str(_cfg_data(cfg, 'heldout_root', '') or '').replace('\\', '/'),
            },
            'tags': list(tags),
            'cache_stats': tag_stats,
        }
        if prev is not None:
            meta['previous_meta'] = {
                'created_at_unix': prev.get('created_at_unix', None),
                'created_at_local': prev.get('created_at_local', None),
                'runtime': prev.get('runtime', None),
                'config': prev.get('config', None),
                'inputs': prev.get('inputs', None),
            }
        _json_dump_atomic(meta, meta_path)
        ddp_print(f'[FeatCache] Wrote meta: {meta_path}')
    if ddp_is_initialized():
        import torch.distributed as dist
        dist.barrier()


def validate_feature_cache_meta_or_warn(cache_dir: str, tags: List[str], cfg: Any, save_dtype: torch.dtype, allow_mismatch: bool):
    meta_path = _feature_cache_meta_path(cache_dir)
    if not os.path.exists(meta_path):
        ddp_print(f'[FeatCache] Meta file not found at {meta_path}. Will write an inferred meta and continue.')
        write_or_update_feature_cache_meta(cache_dir, tags, cfg, save_dtype, device=str(_cfg_runtime(cfg, 'device', 'cpu')), inferred_only=True)
        return
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        ddp_print(f'[FeatCache] Failed to read meta {meta_path}: {e}. Will rewrite inferred meta and continue.')
        write_or_update_feature_cache_meta(cache_dir, tags, cfg, save_dtype, device=str(_cfg_runtime(cfg, 'device', 'cpu')), inferred_only=True)
        return

    problems = []
    cur_ws = get_world_size()
    meta_ws = int(meta.get('runtime', {}).get('world_size', -1))
    if meta_ws != -1 and int(meta_ws) != int(cur_ws):
        problems.append(f'world_size mismatch: meta={meta_ws} current={cur_ws} (per-rank cache would drop data)')
    meta_dtype = str(meta.get('config', {}).get('save_dtype', ''))
    cur_dtype = str(save_dtype).replace('torch.', '')
    if meta_dtype and (meta_dtype != cur_dtype):
        problems.append(f'save_dtype mismatch: meta={meta_dtype} current={cur_dtype}')

    meta_cfg = meta.get('config', {}) or {}
    meta_inputs = meta.get('inputs', {}) or {}

    def _cmp_cfg(key: str, cur_val: Any, fatal: bool = True):
        if key not in meta_cfg:
            return
        mv = meta_cfg.get(key)
        if str(mv) != str(cur_val):
            msg = f'{key} mismatch: meta={mv} current={cur_val}'
            if fatal:
                problems.append(msg)
            else:
                ddp_print(f'[FeatCache][WARN] {msg}')

    def _cmp_input(key: str, cur_val: Any, fatal: bool = True):
        if key not in meta_inputs:
            return
        mv = meta_inputs.get(key)
        if isinstance(mv, dict):
            return
        if str(mv) != str(cur_val):
            msg = f'{key} mismatch: meta={mv} current={cur_val}'
            if fatal:
                problems.append(msg)
            else:
                ddp_print(f'[FeatCache][WARN] {msg}')

    _cmp_cfg('token_key', _cfg_data(cfg, 'token_key', None), fatal=True)
    _cmp_cfg('beh_key', _cfg_data(cfg, 'beh_key', None), fatal=True)
    _cmp_cfg('beh_channels', _cfg_data(cfg, 'beh_channels', None), fatal=True)
    _cmp_cfg('beh_up', _cfg_data(cfg, 'beh_up', None), fatal=True)
    _cmp_cfg('batch_size', _cfg_data(cfg, 'batch_size', None), fatal=False)
    _cmp_cfg('vocab', _cfg_data(cfg, 'vocab', None), fatal=False)
    _cmp_input('heldin_root', str(_cfg_data(cfg, 'heldin_root', '') or '').replace('\\', '/'), fatal=True)
    _cmp_input('heldout_root', str(_cfg_data(cfg, 'heldout_root', '') or '').replace('\\', '/'), fatal=True)

    def _cmp_fp(name: str, cur_fp: dict):
        mfp = meta_inputs.get(name, {})
        if mfp.get('exists', False) and cur_fp.get('exists', False):
            for k in ('size_bytes', 'mtime', 'sha1_head'):
                if (k in mfp) and (k in cur_fp) and (str(mfp[k]) != str(cur_fp[k])):
                    return f'{name} fingerprint mismatch at {k}: meta={mfp.get(k)} current={cur_fp.get(k)}'
        return None

    msg = _cmp_fp('registry_json', _fingerprint_file(_cfg_data(cfg, 'registry_json', '')))
    if msg:
        problems.append(msg)
    msg = _cmp_fp('ar_ckpt', _fingerprint_file(_cfg_backbone(cfg, 'ar_ckpt', '')))
    if msg:
        problems.append(msg)

    cache_stats = meta.get('cache_stats', {})
    for t in tags:
        st = cache_stats.get(t, {})
        if not st:
            problems.append(f"tag '{t}' missing in meta")
        elif not bool(st.get('all_ranks_done', False)):
            problems.append(f"tag '{t}' not complete in meta (all_ranks_done=False). Consider rebuilding or check DONE flags.")

    if len(problems) > 0:
        txt = '\n'.join(['[FeatCache] Meta validation failed:'] + [f'  - {p}' for p in problems])
        if bool(allow_mismatch):
            ddp_print(txt)
            ddp_print('[FeatCache] allow_mismatch=1, continuing anyway.')
        else:
            raise RuntimeError(txt + '\nSet cache.allow_mismatch=true to proceed, or rebuild with cache.force=true / new cache dir.')


class FeatureBatchFileDataset(Dataset):
    def __init__(self, files: List[str]):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], map_location='cpu')


class EpochFileBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, dataset_len: int, shuffle: bool = True, seed: int = 0):
        self.dataset_len = int(dataset_len)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import random
        idxs = list(range(self.dataset_len))
        if self.shuffle and self.dataset_len > 1:
            rng = random.Random(self.seed + self.epoch * 10007 + get_rank() * 1000003)
            rng.shuffle(idxs)
        for i in idxs:
            yield [i]

    def __len__(self):
        return self.dataset_len


def dump_feature_cache_from_loader(
    tag: str,
    backbone,
    encode_fn,
    dl: DataLoader,
    cache_dir: str,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    force: bool = False,
    save_dtype: torch.dtype = torch.float16,
    max_batches: int = -1,
):
    rank = get_rank()
    out_dir = os.path.join(cache_dir, tag)
    _safe_makedirs(out_dir)
    if (not force) and feature_cache_is_done(cache_dir, tag, rank):
        files = list_feature_cache_files(cache_dir, tag, rank)
        if len(files) > 0:
            ddp_print(f'[FeatCache] {tag} rank{rank}: already done, files={len(files)} (skip)')
            return
    ddp_print(f'[FeatCache] {tag} rank{rank}: dumping features -> {out_dir}')
    if force:
        for fp in list_feature_cache_files(cache_dir, tag, rank):
            try:
                os.remove(fp)
            except Exception:
                pass
        try:
            os.remove(_feature_cache_done_flag(cache_dir, tag, rank))
        except Exception:
            pass
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    n_seen = 0
    t0 = time.time()
    for step, batch in enumerate(dl):
        if int(max_batches) > 0 and (step + 1) > int(max_batches):
            break
        xBNT = batch['xBNT']
        lengths = batch['lengths']
        behBCT = batch['behBCT']
        beh_lengths = batch['beh_lengths']
        session_idB = batch['session_idB']
        neuron_offset = int(batch['neuron_offset'][0].item())
        device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(use_amp and device_type == 'cuda')):
            hBNTD, neuron_idsBN = encode_fn(backbone, xBNT, lengths, session_idB, neuron_offset, device)
        pack = {
            'hBNTD': hBNTD.to(dtype=save_dtype).cpu(),
            'lengths': lengths.to(torch.int32).cpu(),
            'session_idB': session_idB.to(torch.int32).cpu(),
            'neuron_offset': batch['neuron_offset'].to(torch.int32).cpu(),
            'trial_idxB': batch['trial_idxB'].to(torch.int32).cpu(),
            'neuron_idsBN': neuron_idsBN.to(torch.int32).cpu(),
            'behBCT': behBCT.to(dtype=save_dtype).cpu(),
            'beh_lengths': beh_lengths.to(torch.int32).cpu(),
        }
        fp = os.path.join(out_dir, f'{tag}_rank{rank}_step{step:06d}.pt')
        torch.save(pack, fp)
        n_seen += 1
    mark_feature_cache_done(cache_dir, tag, rank)
    dt = time.time() - t0
    dt_max = ddp_max_float(dt, device=device)
    ddp_print(f'[FeatCache] {tag} done. rank{rank} batches={n_seen} time={dt_max/60.0:.2f} min')
    if ddp_is_initialized():
        import torch.distributed as dist
        dist.barrier()


def make_feature_cache_loader(cache_dir: str, tag: str, num_workers: int, seed: int, shuffle: bool, allow_empty: bool = False) -> DataLoader:
    rank = get_rank()
    files = list_feature_cache_files(cache_dir, tag, rank)
    if len(files) == 0 and not bool(allow_empty):
        raise FileNotFoundError(f'No feature-cache files for tag={tag} rank={rank} in {cache_dir}')
    ds = FeatureBatchFileDataset(files)
    sampler = EpochFileBatchSampler(len(ds), shuffle=bool(shuffle), seed=int(seed))
    kwargs = dict(
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=lambda xs: xs[0] if (xs is not None and len(xs) > 0) else None,
    )
    if int(num_workers) > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(ds, **kwargs)
