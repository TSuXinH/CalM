from __future__ import annotations

import atexit
import os
import shutil
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


def log_print(msg: str, log_txt_path: Optional[str] = None, force: bool = False) -> None:
    print(msg, force=force)
    if is_main_process() and log_txt_path:
        os.makedirs(os.path.dirname(log_txt_path) or '.', exist_ok=True)
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + "\n")
            f.flush()


def _dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if _dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if _dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    try:
        return int(os.environ.get('LOCAL_RANK', '0'))
    except Exception:
        return 0


def setup_for_distributed(is_master: bool) -> None:
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ddp_setup(device_preference: str = 'cuda', backend: str = 'nccl') -> Tuple[bool, str]:
    if os.environ.get('RANK') is None or os.environ.get('WORLD_SIZE') is None:
        setup_for_distributed(True)
        return False, device_preference

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = get_local_rank()

    if device_preference == 'cpu':
        raise RuntimeError('DDP in this script requires CUDA. Use torchrun with runtime.device=cuda/auto.')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)
    dist.barrier()
    setup_for_distributed(rank == 0)
    atexit.register(ddp_cleanup)
    return True, f'cuda:{local_rank}'


def ddp_cleanup() -> None:
    if _dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


class DistributedNoPadSampler(Sampler[int]):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle: bool = False, seed: int = 0):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            g = np.random.RandomState(self.seed + self.epoch)
            g.shuffle(idx)
        return iter(idx[self.rank:n:self.num_replicas].tolist())

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.num_replicas - 1 - self.rank) // self.num_replicas


def make_distributed_dataloader(dl: DataLoader, shuffle: bool, seed: int, pad: bool = True) -> DataLoader:
    if pad:
        sampler = DistributedSampler(
            dl.dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=False,
        )
    else:
        sampler = DistributedNoPadSampler(
            dl.dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=bool(shuffle),
            seed=int(seed),
        )

    kwargs = dict(
        dataset=dl.dataset,
        batch_size=dl.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=getattr(dl, 'num_workers', 0),
        pin_memory=bool(getattr(dl, 'pin_memory', False)),
        drop_last=bool(getattr(dl, 'drop_last', False)),
        collate_fn=getattr(dl, 'collate_fn', None),
        persistent_workers=getattr(dl, 'persistent_workers', False),
    )
    prefetch_factor = getattr(dl, 'prefetch_factor', None)
    if prefetch_factor is not None and int(getattr(dl, 'num_workers', 0)) > 0:
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(**kwargs)


def get_canonical_heldout_registry(
    registry_json_heldout: Optional[str],
    ckpt_dir: Optional[str],
    base_registry_json: Optional[str],
) -> Optional[str]:
    if registry_json_heldout is not None and str(registry_json_heldout).strip() != '':
        return str(registry_json_heldout)
    if ckpt_dir:
        return os.path.join(str(ckpt_dir), 'registry_heldout.json')
    if base_registry_json:
        return str(base_registry_json) + '.heldout'
    return None


def prepare_canonical_heldout_registry(base_registry_json: Optional[str], heldout_registry: Optional[str]) -> None:
    if not heldout_registry:
        return

    def _do_copy() -> None:
        try:
            os.makedirs(os.path.dirname(heldout_registry) or '.', exist_ok=True)
            shutil.copy2(base_registry_json, heldout_registry)
            print(f'[Registry] seeded heldout registry from base: {base_registry_json} -> {heldout_registry}')
        except Exception as e:
            print(f'[Registry] WARNING: failed to seed heldout registry: {e}')

    if _dist_avail_and_initialized():
        if is_main_process():
            if (not os.path.exists(heldout_registry)) and base_registry_json and os.path.exists(base_registry_json):
                _do_copy()
        dist.barrier()
    else:
        if (not os.path.exists(heldout_registry)) and base_registry_json and os.path.exists(base_registry_json):
            _do_copy()


def ddp_prepare_registry(canonical_registry: Optional[str], per_rank: bool = True) -> Tuple[Optional[str], Callable[[], None]]:
    if not canonical_registry:
        return None, (lambda: None)
    canonical_registry = str(canonical_registry)
    if (not _dist_avail_and_initialized()) or (not per_rank):
        return canonical_registry, (lambda: None)

    rank = get_rank()
    per = canonical_registry + f'.rank{rank}'
    os.makedirs(os.path.dirname(per) or '.', exist_ok=True)
    if not os.path.exists(per):
        if os.path.exists(canonical_registry):
            try:
                shutil.copy2(canonical_registry, per)
            except Exception as e:
                print(f'[Registry] WARNING: copy canonical->per failed: {e}')
        else:
            try:
                with open(per, 'w', encoding='utf-8') as f:
                    f.write('{"paths": []}\n')
            except Exception as e:
                print(f'[Registry] WARNING: create per-rank registry failed: {e}')

    def commit_back() -> None:
        if is_main_process():
            try:
                shutil.copy2(per, canonical_registry)
            except Exception as e:
                print(f'[Registry] WARNING: commit per->canonical failed: {e}')
        dist.barrier()

    return per, commit_back
