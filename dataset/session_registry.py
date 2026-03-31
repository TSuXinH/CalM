from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _to_posix(p: str) -> str:
    return os.path.abspath(p).replace('\\', '/')


def is_under_root(path: str, root: Optional[str]) -> bool:
    if root is None or str(root).strip() == '':
        return False
    p = _to_posix(path)
    r = _to_posix(root)
    return p.startswith(r.rstrip('/') + '/') or p == r.rstrip('/')


def _is_index_dict(d) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    ks = list(d.keys())
    return all(str(k).isdigit() for k in ks)


def _sort_keys(ks):
    return sorted(ks, key=lambda x: int(str(x)) if str(x).isdigit() else str(x))


def _unwrap_object(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _extract_from_trial_obj(trial_obj, npz_path: str, split: str, key: str, alts: Sequence[str]):
    trial_obj = _unwrap_object(trial_obj)
    if isinstance(trial_obj, np.ndarray):
        if trial_obj.ndim >= 2:
            return trial_obj
        if trial_obj.shape == () and isinstance(trial_obj.item(), dict):
            return _extract_from_trial_obj(trial_obj.item(), npz_path, split, key, alts)
    if isinstance(trial_obj, dict):
        if key in trial_obj:
            return trial_obj[key]
        for alt in alts:
            if alt in trial_obj:
                return trial_obj[alt]
        raise KeyError(f'[{npz_path} split={split}] key={key!r} not found. keys={list(trial_obj.keys())[:30]}')
    v = getattr(trial_obj, key, None)
    if v is not None:
        return v
    arr = np.asarray(trial_obj)
    if arr.ndim >= 2:
        return arr
    raise ValueError(f'[{npz_path} split={split}] cannot extract key={key!r} from type={type(trial_obj)}')


def _load_trials_from_split_obj(obj, npz_path: str, split: str,
                                token_key: str, beh_key: str,
                                vocab: Optional[int] = None,
                                beh_channels: Optional[int] = None):
    trials: List[Dict[str, np.ndarray]] = []
    tok_alts = ('tok', 'tokens', 'token_ids', 'tokNT', 'codes', 'ids', 'k')
    beh_alts = ('behavior', 'beh', 'y', 'behCT', 'behaviors')

    def _append(one):
        one = _unwrap_object(one)
        tok = _extract_from_trial_obj(one, npz_path, split, token_key, tok_alts)
        beh = _extract_from_trial_obj(one, npz_path, split, beh_key, beh_alts)
        tok = np.asarray(tok, dtype=np.int64)
        beh = np.asarray(beh, dtype=np.float32)
        if tok.ndim != 2:
            raise ValueError(f'[{npz_path} split={split}] token must be [N,T], got {tok.shape}')
        if beh.ndim != 2:
            raise ValueError(f'[{npz_path} split={split}] behavior must be [C,T], got {beh.shape}')
        if beh_channels is not None and int(beh.shape[0]) != int(beh_channels):
            raise ValueError(f'[{npz_path} split={split}] behavior C mismatch: got {beh.shape[0]} expect {beh_channels}')
        if vocab is not None:
            tmin, tmax = int(tok.min()), int(tok.max())
            if tmin < 0 or tmax >= int(vocab):
                raise ValueError(f'[{npz_path} split={split}] token out of range: min={tmin} max={tmax} vocab={vocab}')
        trials.append({'tokNT': tok, 'behCT': beh})

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == () and isinstance(obj.item(), dict):
            d = obj.item()
            if (token_key not in d) and _is_index_dict(d):
                for k in _sort_keys(d.keys()):
                    _append(d[k])
            else:
                _append(d)
        else:
            for item in obj:
                if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item):
                    for k in _sort_keys(item.keys()):
                        _append(item[k])
                else:
                    _append(item)
    else:
        _append(obj)
    return trials


def load_npz_all_splits(npz_path: str,
                        splits: Sequence[str] = ('train', 'val', 'test'),
                        token_key: str = 'token',
                        beh_key: str = 'behavior',
                        vocab: Optional[int] = None,
                        beh_channels: Optional[int] = None) -> Dict[str, List[Dict[str, np.ndarray]]]:
    out: Dict[str, List[Dict[str, np.ndarray]]] = {}
    with np.load(npz_path, allow_pickle=True) as z:
        for sp in splits:
            if sp not in z.files:
                continue
            out[sp] = _load_trials_from_split_obj(z[sp], npz_path, sp, token_key, beh_key, vocab=vocab, beh_channels=beh_channels)
    return out


def load_registry_paths(registry_json: str) -> List[str]:
    with open(registry_json, 'r', encoding='utf-8') as f:
        reg = json.load(f)
    if isinstance(reg, dict):
        if 'paths' in reg and isinstance(reg['paths'], list):
            return [str(p) for p in reg['paths']]
        if 'npz_list' in reg and isinstance(reg['npz_list'], list):
            return [str(p) for p in reg['npz_list']]
    raise ValueError(f'Unrecognized registry format in {registry_json}. Expect dict with paths or npz_list.')


def build_sessions_from_registry(
    registry_json: str,
    token_key: str,
    beh_key: str,
    vocab: Optional[int],
    beh_channels: int,
    splits: Sequence[str] = ('train', 'val', 'test'),
) -> Tuple[List[dict], dict, Dict[str, Dict[str, List[Dict[str, np.ndarray]]]]]:
    paths = load_registry_paths(registry_json)
    if len(paths) == 0:
        raise FileNotFoundError(f'registry has no paths: {registry_json}')
    cache: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]] = {}
    sessions: List[dict] = []
    neuron_offset = 0
    for sid, p in enumerate(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f'registry path missing: {p}')
        all_trials = load_npz_all_splits(p, splits=splits, token_key=token_key, beh_key=beh_key, vocab=vocab, beh_channels=beh_channels)
        cache[p] = all_trials
        n_train = len(cache[p].get('train', []))
        n_val = len(cache[p].get('val', []))
        n_test = len(cache[p].get('test', []))
        n_neurons = None
        for sp in ('train', 'val', 'test'):
            if len(cache[p].get(sp, [])) > 0:
                n_neurons = int(cache[p][sp][0]['tokNT'].shape[0])
                break
        if n_neurons is None:
            raise ValueError(f'No trials found in any split for {p}')
        sessions.append(dict(
            path=p,
            session_id=int(sid),
            neuron_offset=int(neuron_offset),
            n_neurons=int(n_neurons),
            n_train=int(n_train),
            n_val=int(n_val),
            n_test=int(n_test),
        ))
        neuron_offset += int(n_neurons)
    meta = dict(
        n_sessions=len(sessions),
        n_neurons_total=int(neuron_offset),
        sessions=sessions,
        paths=paths,
    )
    return sessions, meta, cache


def filter_sessions_by_root(sessions: List[dict], root: Optional[str]) -> List[int]:
    if root is None or str(root).strip() == '':
        return list(range(len(sessions)))
    return [int(s['session_id']) for s in sessions if is_under_root(s['path'], root)]


def _fingerprint_file(path: str, head_bytes: int = 1024 * 1024) -> dict:
    if path is None or str(path).strip() == '':
        return {'path': '', 'exists': False}
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        return {'path': ap, 'exists': False}
    st = os.stat(ap)
    h = hashlib.sha1()
    try:
        with open(ap, 'rb') as f:
            h.update(f.read(int(head_bytes)))
        sha1_head = h.hexdigest()
    except Exception:
        sha1_head = ''
    return {
        'path': ap.replace('\\', '/'),
        'exists': True,
        'size_bytes': int(st.st_size),
        'mtime': float(st.st_mtime),
        'sha1_head': sha1_head,
    }


def build_offsets_from_registry(registry_json: str, token_key: str) -> Tuple[List[str], List[int], List[int]]:
    paths = load_registry_paths(registry_json)
    n_neurons = []
    offsets = []
    off = 0
    tok_alts = ('tok', 'tokens', 'token_ids', 'tokNT', 'codes', 'ids', 'k')
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            tokNT = None
            for sp in ('train', 'val', 'test'):
                if sp not in z.files:
                    continue
                obj = z[sp]
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    if obj.shape == () and isinstance(obj.item(), dict):
                        d = obj.item()
                        if (token_key not in d) and _is_index_dict(d):
                            k0 = _sort_keys(d.keys())[0]
                            tokNT = _extract_from_trial_obj(d[k0], p, sp, token_key, tok_alts)
                        else:
                            tokNT = _extract_from_trial_obj(d, p, sp, token_key, tok_alts)
                    else:
                        item0 = obj.flat[0]
                        if isinstance(item0, dict) and (token_key not in item0) and _is_index_dict(item0):
                            k0 = _sort_keys(item0.keys())[0]
                            tokNT = _extract_from_trial_obj(item0[k0], p, sp, token_key, tok_alts)
                        else:
                            tokNT = _extract_from_trial_obj(item0, p, sp, token_key, tok_alts)
                else:
                    tokNT = _extract_from_trial_obj(obj, p, sp, token_key, tok_alts)
                break
        tokNT = np.asarray(tokNT)
        N = int(tokNT.shape[0])
        offsets.append(int(off))
        n_neurons.append(int(N))
        off += N
    return paths, offsets, n_neurons
