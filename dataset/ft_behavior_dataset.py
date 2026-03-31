from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model.behavior_decoder.distributed import get_rank, get_world_size


class MultiSessionTrialDatasetCached(Dataset):
    def __init__(self, sessions: List[dict], cache: Dict[str, dict], split: str, selected_session_ids: Optional[Sequence[int]] = None):
        assert split in ('train', 'val', 'test')
        self.sessions = sessions
        self.cache = cache
        self.split = split
        sel = None if selected_session_ids is None else set(int(x) for x in selected_session_ids)
        self.refs: List[dict] = []
        for s in sessions:
            sid = int(s['session_id'])
            if sel is not None and sid not in sel:
                continue
            n_trials = int(s.get(f'n_{split}', 0))
            for ti in range(n_trials):
                self.refs.append(dict(
                    path=s['path'],
                    session_id=sid,
                    neuron_offset=int(s['neuron_offset']),
                    trial_idx=int(ti),
                ))

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx: int):
        r = self.refs[idx]
        trials = self.cache[r['path']].get(self.split, None)
        if trials is None:
            raise KeyError(f'split {self.split} missing in cache for {r["path"]}')
        tr = trials[r['trial_idx']]
        tokNT = torch.from_numpy(tr['tokNT']).long()
        behCT = torch.from_numpy(tr['behCT']).float()
        return {
            'tokNT': tokNT,
            'behCT': behCT,
            'length': int(tokNT.shape[1]),
            'session_id': int(r['session_id']),
            'neuron_offset': int(r['neuron_offset']),
            'trial_idx': int(r['trial_idx']),
        }


class SessionBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: MultiSessionTrialDatasetCached, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.sid2idx: Dict[int, List[int]] = {}
        for i, r in enumerate(dataset.refs):
            sid = int(r['session_id'])
            self.sid2idx.setdefault(sid, []).append(i)
        self.session_ids = sorted(self.sid2idx.keys())
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 10007)
        sids = self.session_ids[:]
        if self.shuffle:
            rng.shuffle(sids)
        for sid in sids:
            idxs = self.sid2idx[sid][:]
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for sid in self.session_ids:
            n = len(self.sid2idx[sid])
            total += n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return total


class DDPSessionBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: MultiSessionTrialDatasetCached,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        pad_to_world_size: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas) if num_replicas is not None else get_world_size()
        self.rank = int(rank) if rank is not None else get_rank()
        self.pad_to_world_size = bool(pad_to_world_size)
        self.epoch = 0
        self.sid2idx: Dict[int, List[int]] = {}
        for i, r in enumerate(dataset.refs):
            sid = int(r['session_id'])
            self.sid2idx.setdefault(sid, []).append(i)
        self.session_ids = sorted(self.sid2idx.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _build_all_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch * 10007)
        sids = self.session_ids[:]
        if self.shuffle:
            rng.shuffle(sids)
        all_batches: List[List[int]] = []
        for sid in sids:
            idxs = self.sid2idx[sid][:]
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.pad_to_world_size and self.num_replicas > 1 and len(all_batches) > 0:
            rem = len(all_batches) % self.num_replicas
            if rem != 0:
                pad = self.num_replicas - rem
                all_batches.extend(all_batches[:pad])
        return all_batches

    def __iter__(self):
        all_batches = self._build_all_batches()
        if self.num_replicas <= 1:
            yield from all_batches
            return
        for b in all_batches[self.rank::self.num_replicas]:
            yield b

    def __len__(self):
        all_batches = self._build_all_batches()
        if self.num_replicas <= 1:
            return len(all_batches)
        return (len(all_batches) - self.rank + self.num_replicas - 1) // self.num_replicas


def collate_padT_behavior(batch: List[dict], beh_channels: int, beh_up: int, pad_token: int = 0):
    B = len(batch)
    sid0 = int(batch[0]['session_id'])
    N0 = int(batch[0]['tokNT'].shape[0])
    for b in batch:
        assert int(b['session_id']) == sid0, 'Batch must share session_id'
        assert int(b['tokNT'].shape[0]) == N0, 'Batch must share N'
    Ttoks = [int(b['tokNT'].shape[1]) for b in batch]
    Tmax = int(max(Ttoks))
    xBNT = torch.full((B, N0, Tmax), int(pad_token), dtype=torch.long)
    lengths = torch.tensor(Ttoks, dtype=torch.long)
    BehTmax = Tmax * int(beh_up)
    behBCT = torch.zeros((B, int(beh_channels), BehTmax), dtype=torch.float32)
    beh_lengths = torch.zeros((B,), dtype=torch.long)
    neuron_offset = int(batch[0]['neuron_offset'])
    session_idB = torch.tensor([sid0] * B, dtype=torch.long)
    trial_idxB = torch.tensor([int(b['trial_idx']) for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        tok = b['tokNT']
        T = int(tok.shape[1])
        xBNT[i, :, :T] = tok
        beh = b['behCT']
        Tb = int(beh.shape[1])
        target_Tb = T * int(beh_up)
        use = min(Tb, target_Tb)
        behBCT[i, :, :use] = beh[:, :use]
        beh_lengths[i] = int(use)
    return {
        'xBNT': xBNT,
        'lengths': lengths,
        'behBCT': behBCT,
        'beh_lengths': beh_lengths,
        'session_idB': session_idB,
        'neuron_offset': torch.tensor([neuron_offset] * B, dtype=torch.long),
        'trial_idxB': trial_idxB,
    }


def make_loader(
    sessions: List[dict],
    cache: dict,
    split: str,
    selected_sids: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    beh_channels: int,
    beh_up: int,
    num_workers: int = 0,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    pad_to_world_size: bool = True,
):
    ds = MultiSessionTrialDatasetCached(sessions=sessions, cache=cache, split=split, selected_session_ids=selected_sids)
    if ddp and int(world_size) > 1:
        sampler = DDPSessionBatchSampler(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
            num_replicas=int(world_size),
            rank=int(rank),
            pad_to_world_size=bool(pad_to_world_size),
        )
    else:
        sampler = SessionBatchSampler(ds, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=False)
    dl_kwargs = dict(
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=lambda b: collate_padT_behavior(b, beh_channels=beh_channels, beh_up=beh_up, pad_token=0),
    )
    if int(num_workers) > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(ds, **dl_kwargs)
