import os
import glob
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def _stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _is_index_dict(d) -> bool:
    if not isinstance(d, dict) or len(d) == 0: return False
    ks = list(d.keys())
    return all(str(k).isdigit() for k in ks)

def _sort_keys(ks):
    return sorted(ks, key=lambda x: int(str(x)) if str(x).isdigit() else str(x))

def _extract_tok_from_trial_obj(trial_obj, npz_path: str, split: str, token_key: str):
    if isinstance(trial_obj, np.ndarray):
        if trial_obj.ndim >= 2: return trial_obj
        if trial_obj.shape == () and isinstance(trial_obj.item(), dict): return _extract_tok_from_trial_obj(trial_obj.item(), npz_path, split, token_key)
    if isinstance(trial_obj, dict):
        if token_key in trial_obj:
            return trial_obj[token_key]
        for alt in ("tok", "tokens", "token_ids", "tokNT", "codes", "ids", "k"):
            if alt in trial_obj: return trial_obj[alt]
        raise KeyError(
            f"[load_npz_tokens_for_splits] token_key='{token_key}' not found in trial dict "
            f"({npz_path} split={split}). keys={list(trial_obj.keys())[:30]}"
        )
    tok = getattr(trial_obj, token_key, None)
    if tok is not None: return tok
    arr = np.asarray(trial_obj)
    if arr.ndim >= 2: return arr
    raise ValueError(
        f"[load_npz_tokens_for_splits] Cannot extract tokens from type={type(trial_obj)} "
        f"({npz_path} split={split})."
    )

def _count_trials_in_split_obj(obj, token_key: str) -> int:
    n = 0
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == () and isinstance(obj.item(), dict):
            d = obj.item()
            if (token_key not in d) and _is_index_dict(d): n += len(d.keys())
            else: n += 1
        else:
            for item in obj:
                if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item): n += len(item.keys())
                else: n += 1
        return int(n)
    return 1

def _peek_first_tok_shape(obj, token_key: str, npz_path: str, split: str) -> Optional[Tuple[int, int]]:
    try:
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == () and isinstance(obj.item(), dict):
            d = obj.item()
            if (token_key not in d) and _is_index_dict(d):
                k0 = _sort_keys(d.keys())[0]
                tok = _extract_tok_from_trial_obj(d[k0], npz_path, split, token_key)
            else: tok = _extract_tok_from_trial_obj(d, npz_path, split, token_key)
            tok = np.asarray(tok)
            if tok.ndim >= 2: return int(tok.shape[0]), int(tok.shape[1])
            return None
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            if obj.size == 0: return None
            item0 = obj.flat[0]
            if isinstance(item0, dict) and (token_key not in item0) and _is_index_dict(item0):
                k0 = _sort_keys(item0.keys())[0]
                tok = _extract_tok_from_trial_obj(item0[k0], npz_path, split, token_key)
            else: tok = _extract_tok_from_trial_obj(item0, npz_path, split, token_key)
            tok = np.asarray(tok)
            if tok.ndim >= 2: return int(tok.shape[0]), int(tok.shape[1])
            return None
        tok = _extract_tok_from_trial_obj(obj, npz_path, split, token_key)
        tok = np.asarray(tok)
        if tok.ndim >= 2: return int(tok.shape[0]), int(tok.shape[1])
        return None
    except Exception: return None

def load_npz_tokens_for_splits(
    npz_path: str,
    splits: Sequence[str] = ("train", "val", "test"),
    token_key: str = "token",
    vocab: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    with np.load(npz_path, allow_pickle=True) as z:
        for split in splits:
            if split not in z: continue
            obj = z[split]
            trials: List[np.ndarray] = []
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                if obj.shape == () and isinstance(obj.item(), dict):
                    d = obj.item()
                    if (token_key not in d) and _is_index_dict(d):
                        for k in _sort_keys(d.keys()):
                            tok = _extract_tok_from_trial_obj(d[k], npz_path, split, token_key)
                            trials.append(np.asarray(tok, dtype=np.int64))
                    else:
                        tok = _extract_tok_from_trial_obj(d, npz_path, split, token_key)
                        trials.append(np.asarray(tok, dtype=np.int64))
                else:
                    for item in obj:
                        if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item):
                            for k in _sort_keys(item.keys()):
                                tok = _extract_tok_from_trial_obj(item[k], npz_path, split, token_key)
                                trials.append(np.asarray(tok, dtype=np.int64))
                        else:
                            tok = _extract_tok_from_trial_obj(item, npz_path, split, token_key)
                            trials.append(np.asarray(tok, dtype=np.int64))
            else:
                tok = _extract_tok_from_trial_obj(obj, npz_path, split, token_key)
                trials.append(np.asarray(tok, dtype=np.int64))
            if vocab is not None and len(trials) > 0:
                for ti, t in enumerate(trials):
                    tmin, tmax = int(t.min()), int(t.max())
                    if tmin < 0 or tmax >= int(vocab):
                        raise ValueError(
                            f"[{npz_path} split={split}] token out of range at trial#{ti}: "
                            f"min={tmin} max={tmax} vocab={vocab}"
                        )
            out[split] = trials
    return out

class MultiSessionTrialDatasetV2(Dataset):
    def __init__(
        self,
        sessions: List[dict],
        split: str,
        token_key: str = "token",
        vocab: Optional[int] = None,
        preload: bool = False,
        cache: Optional[Dict[str, Dict[str, List[np.ndarray]]]] = None,
    ):
        assert split in ("train", "val", "test")
        self.sessions = sessions
        self.split = split
        self.token_key = token_key
        self.vocab = vocab
        self.preload = preload
        self._shared_cache = cache  # could be None
        self._local_cache: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.refs = []  # {path, session_id, neuron_offset, trial_idx}
        for s in sessions:
            n_trials = int(s.get(f"n_{split}", 0))
            for ti in range(n_trials):
                self.refs.append(dict(
                    path=s["path"],
                    session_id=int(s["session_id"]),
                    neuron_offset=int(s["neuron_offset"]),
                    trial_idx=int(ti),
                ))

    def __len__(self):
        return len(self.refs)

    def _get_tokNT(self, path: str, split: str, trial_idx: int) -> np.ndarray:
        if self._shared_cache is not None and path in self._shared_cache:
            return self._shared_cache[path][split][trial_idx]
        if path not in self._local_cache:
            self._local_cache[path] = load_npz_tokens_for_splits(
                path, splits=(split,), token_key=self.token_key, vocab=self.vocab
            )
        return self._local_cache[path][split][trial_idx]

    def __getitem__(self, idx: int):
        r = self.refs[idx]
        tokNT = self._get_tokNT(r["path"], self.split, r["trial_idx"])
        tokNT = np.asarray(tokNT, dtype=np.int64)
        return {
            "tokNT": torch.from_numpy(tokNT),  # (N,T)
            "length": int(tokNT.shape[1]),
            "N": int(tokNT.shape[0]),
            "session_id": int(r["session_id"]),
            "neuron_offset": int(r["neuron_offset"]),
            "trial_idx": int(r["trial_idx"]),
        }

def collate_fixedN_padT_v2(
    batch: List[dict],
    n_sub: int,
    pad_token: int = 0,
    deterministic: bool = False,
    base_seed: int = 0,
):
    B = len(batch)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    Tmax = int(lengths.max().item())
    if n_sub <= 0:
        if B != 1: raise ValueError("[collate] n_sub<=0 (all neurons) requires batch_size=1 (B==1).")
        n_sub_eff = int(batch[0]["N"])
    else: n_sub_eff = int(n_sub)
    xBNT = torch.full((B, n_sub_eff, Tmax), pad_token, dtype=torch.long)
    session_idB = torch.tensor([b["session_id"] for b in batch], dtype=torch.long)
    neuron_idsBN = torch.empty((B, n_sub_eff), dtype=torch.long)
    trial_idxB = torch.tensor([b["trial_idx"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        tokNT = b["tokNT"]  # torch tensor [N,T]
        N, T = tokNT.shape
        if n_sub <= 0:
            xBNT[i, :, :T] = tokNT
            neuron_idsBN[i] = int(b["neuron_offset"]) + torch.arange(N, dtype=torch.long)
        else:
            if deterministic:
                seed_i = int(base_seed) ^ _stable_int_hash(f"{int(b['session_id'])}_{int(b['trial_idx'])}")
                g = torch.Generator()
                g.manual_seed(seed_i)
                if N >= n_sub_eff: idx = torch.randperm(N, generator=g)[:n_sub_eff]
                else: idx = torch.randint(0, N, (n_sub_eff,), generator=g)
            else:
                if N >= n_sub_eff: idx = torch.randperm(N)[:n_sub_eff]
                else: idx = torch.randint(0, N, (n_sub_eff,))

            sub = tokNT.index_select(0, idx)  # [n_sub, T]
            xBNT[i, :, :T] = sub
            neuron_idsBN[i] = int(b["neuron_offset"]) + idx
    return {
        "xBNT": xBNT,
        "lengths": lengths,
        "session_idB": session_idB,
        "neuron_idsBN": neuron_idsBN,
        "trial_idxB": trial_idxB,
    }


def _load_trials_from_split_obj(obj, npz_path: str, split: str, token_key: str, vocab: Optional[int]):
    trials: List[np.ndarray] = []
    def _append(tok):
        arr = np.asarray(tok, dtype=np.int64)
        trials.append(arr)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == () and isinstance(obj.item(), dict):
            d = obj.item()
            if (token_key not in d) and _is_index_dict(d):
                for k in _sort_keys(d.keys()):
                    tok = _extract_tok_from_trial_obj(d[k], npz_path, split, token_key)
                    _append(tok)
            else:
                tok = _extract_tok_from_trial_obj(d, npz_path, split, token_key)
                _append(tok)
        else:
            for item in obj:
                if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item):
                    for k in _sort_keys(item.keys()):
                        tok = _extract_tok_from_trial_obj(item[k], npz_path, split, token_key)
                        _append(tok)
                else:
                    tok = _extract_tok_from_trial_obj(item, npz_path, split, token_key)
                    _append(tok)
    else:
        tok = _extract_tok_from_trial_obj(obj, npz_path, split, token_key)
        _append(tok)
    if vocab is not None and len(trials) > 0:
        V = int(vocab)
        for ti, t in enumerate(trials):
            tmin, tmax = int(t.min()), int(t.max())
            if tmin < 0 or tmax >= V:
                raise ValueError(
                    f"[{npz_path} split={split}] token out of range at trial#{ti}: "
                    f"min={tmin} max={tmax} vocab={V}"
                )
    return trials

def build_session_index_v2(
    data_root: str,
    pattern: str = "*.npz",
    token_key: str = "token",
    vocab: Optional[int] = None,
    exclude_dir_prefix=("prev_",),
    registry_json: Optional[str] = None,
    preload: bool = False,
    preload_splits: Sequence[str] = ("train", "val", "test"),
):
    print("[Index] Build session index.")
    paths = sorted(glob.glob(os.path.join(data_root, "**", pattern), recursive=True))
    npz_list = []
    for p in paths:
        parent = os.path.basename(os.path.dirname(p))
        if any(parent.startswith(pref) for pref in exclude_dir_prefix): continue
        npz_list.append(p)
    if len(npz_list) == 0: raise FileNotFoundError(f"No npz found under {data_root} pattern={pattern}")
    if registry_json:
        if os.path.exists(registry_json):
            with open(registry_json, "r") as f: registry = json.load(f)
        else: registry = {"paths": []}
        known = set(registry.get("paths", []))
        for p in npz_list:
            if p not in known: registry.setdefault("paths", []).append(p)
        os.makedirs(os.path.dirname(registry_json) or ".", exist_ok=True)
        with open(registry_json, "w") as f: json.dump(registry, f, indent=2)
        ordered = [p for p in registry.get("paths", []) if p in set(npz_list)]
        npz_list = ordered
    sessions = []
    neuron_offset = 0
    shared_cache = {} if preload else None
    for sid, p in enumerate(npz_list):
        print("[PATH]:", p)
        n_train = n_val = n_test = 0
        n_neurons = 0
        cache_entry = {} if preload else None
        with np.load(p, allow_pickle=True) as z:
            for split in ("train", "val", "test"):
                if split not in z: continue
                obj = z[split]
                if preload and (split in preload_splits):
                    trials = _load_trials_from_split_obj(obj, p, split, token_key, vocab)
                    cache_entry[split] = trials
                    cnt = len(trials)
                    if n_neurons == 0 and cnt > 0:
                        t0 = trials[0]
                        if t0.ndim >= 2: n_neurons = int(t0.shape[0])
                else:
                    cnt = _count_trials_in_split_obj(obj, token_key=token_key)
                    if n_neurons == 0:
                        shp = _peek_first_tok_shape(obj, token_key=token_key, npz_path=p, split=split)
                        if shp is not None: n_neurons = int(shp[0])
                if split == "train": n_train = int(cnt)
                elif split == "val": n_val = int(cnt)
                else: n_test = int(cnt)
        if preload:
            for sp in preload_splits: cache_entry.setdefault(sp, [])
            shared_cache[p] = cache_entry
        sessions.append(dict(
            path=p,
            session_id=int(sid),
            neuron_offset=int(neuron_offset),
            n_neurons=int(n_neurons),
            n_train=int(n_train),
            n_val=int(n_val),
            n_test=int(n_test),
        ))
        neuron_offset += n_neurons
    meta = dict(
        n_sessions=len(sessions),
        n_neurons_total=int(neuron_offset),
        sessions=sessions,
        npz_list=npz_list,
    )
    if preload: return sessions, meta, shared_cache
    return sessions, meta

def build_multisession_loaders_fixedN_v2(
    data_root: str,
    n_sub: int,
    pattern: str = "*.npz",
    token_key: str = "token",
    vocab: int = 128,
    batch_size: int = 8,
    eval_batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    preload: bool = False,
    exclude_dir_prefix=("prev_",),
    registry_json: Optional[str] = None,
    base_seed: int = 0,
    preload_splits: Sequence[str] = ("train", "val", "test"),
    eval_all_neurons: bool = True,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
):
    if preload:
        sessions, meta, shared_cache = build_session_index_v2(
            data_root=data_root,
            pattern=pattern,
            token_key=token_key,
            vocab=vocab,
            exclude_dir_prefix=exclude_dir_prefix,
            registry_json=registry_json,
            preload=True,
            preload_splits=preload_splits,
        )
        num_workers = 0
    else:
        sessions, meta = build_session_index_v2(
            data_root=data_root,
            pattern=pattern,
            token_key=token_key,
            vocab=vocab,
            exclude_dir_prefix=exclude_dir_prefix,
            registry_json=registry_json,
            preload=False,
        )
        shared_cache = None
    def _mk(split: str, shuffle: bool, bs: int, det: bool):
        ds = MultiSessionTrialDatasetV2(
            sessions=sessions,
            split=split,
            token_key=token_key,
            vocab=vocab,
            preload=preload,
            cache=shared_cache,
        )
        n_sub_used = (-1 if (eval_all_neurons and split in ("val","test")) else n_sub)
        if (n_sub_used <= 0) and (bs != 1): raise ValueError("[Loader] eval_all_neurons requires eval_batch_size=1.")
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=lambda b: collate_fixedN_padT_v2(
                b, n_sub=n_sub_used, pad_token=0, deterministic=det, base_seed=base_seed
            ),
        )
    dl_tr = _mk(train_split, shuffle=True,  bs=batch_size,      det=False)
    dl_va = _mk(val_split,   shuffle=False, bs=eval_batch_size, det=True)
    dl_te = _mk(test_split,  shuffle=False, bs=eval_batch_size, det=True)
    return dl_tr, dl_va, dl_te, meta

class TokenTraceDataset(Dataset):
    def __init__(self, trials: List[np.ndarray]):
        self.trials = [torch.from_numpy(x.astype(np.int64)) for x in trials]
        N0 = self.trials[0].shape[0]
        assert all(x.ndim==2 and x.shape[0]==N0 for x in self.trials), "All trials must be [N,T] with same N."
    def __len__(self): return len(self.trials)
    def __getitem__(self, idx): return self.trials[idx]    

@torch.no_grad()
def greedy_rollout_tokens_one(
    model,
    tok_gt_NT: torch.Tensor,          # [N,T] long
    neuron_ids_N: torch.Tensor,       # [N] long (global neuron ids)
    session_id: torch.Tensor,         # scalar tensor
    prefix_tokens: int,
    device: str = "cuda",
):
    N, T = tok_gt_NT.shape
    if T <= 1: return tok_gt_NT.clone()
    P = int(max(1, min(prefix_tokens, T - 1)))   # keep at least 1 token
    gen = tok_gt_NT[:, :P].clone()               # [N,P]
    neuron_ids_N = neuron_ids_N.to(device)
    sid = session_id.reshape(1).to(device)
    for _ in range(P, T):
        curT = gen.shape[1]                      # context length
        x_in = gen.unsqueeze(0)                  # [1,N,curT]
        time_pos = torch.arange(curT, device=device).unsqueeze(0)  # [1,curT]
        kpmask = torch.ones((1, curT), dtype=torch.bool, device=device)
        neuron_idsBNT = neuron_ids_N.view(1, N, 1).expand(1, N, curT)  # [1,N,curT]
        logits = model(x_in, neuron_idsBNT, time_pos, kpmask, session_idB=sid)  # [1,N,curT,V]
        nxt = logits[0, :, -1].argmax(dim=-1)     # [N]
        gen = torch.cat([gen, nxt.unsqueeze(1)], dim=1)
    return gen


class MultiSessionTrialDataset(Dataset):
    def __init__(self, sessions, split: str, token_key="token", vocab: int = None, preload=False):
        assert split in ("train", "val", "test")
        self.sessions = sessions
        self.split = split
        self.token_key = token_key
        self.vocab = vocab
        self.preload = preload
        self.refs = []  # list of dict: {path, session_id, neuron_offset, trial_idx, n_neurons}
        self._cache = {}  # if preload: path -> splits dict
        for s in sessions:
            if preload:
                splits = load_npz_splits_tokens(s["path"], token_key=token_key, vocab=vocab)
                self._cache[s["path"]] = splits
                n_trials = len(splits[split])
            else:
                splits = load_npz_splits_tokens(s["path"], token_key=token_key, vocab=vocab)
                n_trials = len(splits[split])
            for ti in range(n_trials):
                self.refs.append({
                    "path": s["path"],
                    "session_id": s["session_id"],
                    "neuron_offset": s["neuron_offset"],
                    "n_neurons": s["n_neurons"],
                    "trial_idx": ti,
                })

    def __len__(self): return len(self.refs)

    def __getitem__(self, idx):
        r = self.refs[idx]
        if self.preload: tokNT = self._cache[r["path"]][self.split][r["trial_idx"]]
        else:
            splits = load_npz_splits_tokens(r["path"], token_key=self.token_key, vocab=self.vocab)
            tokNT = splits[self.split][r["trial_idx"]]
        tokNT = np.asarray(tokNT, dtype=np.int64)  # [N,T]
        return {"tokNT": torch.from_numpy(tokNT), "length": int(tokNT.shape[1]), "N": int(tokNT.shape[0]), "session_id": int(r["session_id"]), "neuron_offset": int(r["neuron_offset"])}
    