from __future__ import annotations
import re
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader

def collate_fn_Tseng_trial(batch):
    if len(batch) != 1: raise ValueError(f"collate_fn_Tseng_trial expects batch_size=1 to avoid cross-trial padding, got {len(batch)}.")
    batch_process = batch[0]
    return torch.unsqueeze(batch_process, dim=1)

class DatasetPredefinedTsengTrial(Dataset):
    def __init__(self, Tseng_data_padding, lengths, repeat_factor, sample_batch):
        super().__init__()
        self.dataset = {}
        self.repeat_factor = repeat_factor
        self.sample_batch = sample_batch
        for idx, trial_data_raw in enumerate(Tseng_data_padding):
            trial_length = lengths[idx]
            trial_data_raw = trial_data_raw[:, :trial_length]
            remainder = trial_data_raw.shape[1] % 4
            if remainder == 1: trial_data = trial_data_raw[:, :-1]
            elif remainder == 2:
                last_frame = trial_data_raw[:, -1:]
                trial_data = np.concatenate([trial_data_raw, np.repeat(last_frame, repeats=2, axis=1)], axis=1)
            elif remainder == 3:
                last_frame = trial_data_raw[:, -1:]
                trial_data = np.concatenate([trial_data_raw, last_frame], axis=1)
            else: trial_data = trial_data_raw
            self.dataset[idx] = trial_data
        
    def __len__(self):
        return len(self.dataset) * self.repeat_factor
    
    def __getitem__(self, idx):
        true_idx = idx % len(self.dataset)
        sample_data = self.dataset[true_idx]
        selected_neurons = torch.randperm(sample_data.shape[0])[: self.sample_batch].numpy()
        return torch.from_numpy(self.dataset[true_idx][selected_neurons]).float()
    
def find_npz_files(
    root_dir: str,
    pattern: str = "*.npz",
    exclude_dir_prefix: Tuple[str, ...] = ("prev_",),
) -> List[str]:
    root = Path(root_dir)
    npz_paths: List[str] = []
    for p in root.rglob(pattern):
        if not p.is_file(): continue
        parts = [pp.name for pp in p.parents]
        if any(part.startswith(exclude_dir_prefix) for part in parts): continue
        print(p)
        npz_paths.append(str(p))
    npz_paths.sort()
    return npz_paths

def _align_length_if_needed(trial_x: np.ndarray, align_mod: Optional[int]) -> np.ndarray:
    x = np.asarray(trial_x)
    time_axis = -1
    T = x.shape[time_axis]
    if T == 0: return x
    r = T % align_mod
    if r == 0: return x
    if r == 1:
        slicer = [slice(None)] * x.ndim
        slicer[time_axis] = slice(0, T - 1)
        return x[tuple(slicer)]
    pad = align_mod - r
    slicer_last = [slice(None)] * x.ndim
    slicer_last[time_axis] = slice(T - 1, T)     # keep dims
    last = x[tuple(slicer_last)]                 # shape keeps time dim = 1
    reps = [1] * x.ndim
    reps[time_axis] = pad
    pad_block = np.tile(last, reps)
    return np.concatenate([x, pad_block], axis=time_axis)

class MultiNpzTsengTrialDataset(Dataset):
    def __init__(
        self,
        npz_paths: List[str],
        split: str = "train",
        sample_batch: int = 32,
        repeat_factor: int = 1,
        align_mod: Optional[int] = 4,  
        preload: bool = True,       
        dtype: np.dtype = np.float32,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        self.npz_paths = list(npz_paths)
        self.split = split
        self.sample_batch = int(sample_batch)
        self.repeat_factor = int(repeat_factor)
        self.align_mod = align_mod
        self.preload = bool(preload)
        self.dtype = dtype
        self._trials: Optional[List[np.ndarray]] = None
        self._index: List[Tuple[str, int]] = []
        self._lengths_cache: Dict[str, np.ndarray] = {}
        self._xkey = f"{split}_padded_X"
        self._lkey = f"{split}_lengths"
        if self.preload:
            self._trials = []
            for path in self.npz_paths:
                with np.load(path) as d:
                    X = d[self._xkey]            # [n_trial, N, T_max]
                    L = d[self._lkey].astype(np.int64)
                n_trial = len(L)
                for i in range(n_trial):
                    Ti = int(L[i])
                    if Ti <= 0: continue
                    trial = np.asarray(X[i], dtype=self.dtype)[:, :Ti]  # [N, T]
                    trial = _align_length_if_needed(trial, self.align_mod)
                    self._trials.append(trial)
            if len(self._trials) == 0: raise RuntimeError(f"No trials found for split={split} in given npz paths.")
        else:
            for path in self.npz_paths:
                with np.load(path) as d: L = d[self._lkey].astype(np.int64)
                self._lengths_cache[path] = L
                for i, Ti in enumerate(L):
                    if int(Ti) <= 0: continue
                    self._index.append((path, i))
            if len(self._index) == 0: raise RuntimeError(f"No trials found for split={split} in given npz paths.")

    def __len__(self) -> int:
        base = len(self._trials) if self.preload else len(self._index)
        return base * self.repeat_factor

    def _get_trial_np(self, base_idx: int) -> np.ndarray:
        if self.preload:
            assert self._trials is not None
            return self._trials[base_idx]
        path, local_i = self._index[base_idx]
        with np.load(path) as d:
            X = d[self._xkey]
            L = self._lengths_cache.get(path)
            if L is None:
                L = d[self._lkey].astype(np.int64)
                self._lengths_cache[path] = L
            Ti = int(L[local_i])
            trial = np.asarray(X[local_i], dtype=self.dtype)[:, :Ti]
            trial = _align_length_if_needed(trial, self.align_mod)
            return trial

    def __getitem__(self, idx: int) -> torch.Tensor:
        base = len(self._trials) if self.preload else len(self._index)
        true_idx = int(idx % base)
        trial = self._get_trial_np(true_idx)     # [N, T]
        N = trial.shape[0]
        perm = torch.randperm(N)
        sel = perm[: self.sample_batch]
        out = torch.from_numpy(trial[sel.numpy()]).float()  # [B_neuron, T]
        return out

def collate_trial_neurons(batch: List[torch.Tensor]) -> torch.Tensor:
    if len(batch) != 1: raise ValueError(f"Expected DataLoader batch_size=1 (to avoid cross-trial padding), but got {len(batch)}.")
    x = batch[0]                      # [B_neuron, T]
    return x.unsqueeze(1)             # [B_neuron, 1, T]

def build_multisession_loaders(
    data_root: str,
    pattern: str = "*.npz",
    exclude_dir_prefix: Tuple[str, ...] = ("prev_",),
    sample_batch: int = 32,
    batch_size: int = 1,           
    num_workers: int = 4,
    repeat_factor: int = 1,
    align_mod: Optional[int] = 4, 
    preload: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    npz_paths = find_npz_files(data_root, pattern=pattern, exclude_dir_prefix=exclude_dir_prefix)
    if len(npz_paths) == 0: raise RuntimeError(f"No npz files found under {data_root}")
    train_ds = MultiNpzTsengTrialDataset(
        npz_paths=npz_paths,
        split="train",
        sample_batch=sample_batch,
        repeat_factor=repeat_factor,
        align_mod=align_mod,
        preload=preload,
    )
    val_ds = MultiNpzTsengTrialDataset(
        npz_paths=npz_paths,
        split="val",
        sample_batch=sample_batch,
        repeat_factor=1,
        align_mod=align_mod,
        preload=preload,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_trial_neurons,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_trial_neurons,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader, npz_paths
