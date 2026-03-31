from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def ddp_max_float(x: float, device: str) -> float:
    if not ddp_is_initialized():
        return float(x)
    dev = 'cpu' if (device is None or str(device).startswith('cpu')) else str(device)
    t = torch.tensor([float(x)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def setup_for_distributed(is_master: bool):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ddp_setup(runtime, preferred_device: str = 'cuda') -> tuple[bool, str, int, int]:
    use_ddp = bool(getattr(runtime, 'ddp', False)) or (int(os.environ.get('WORLD_SIZE', '1')) > 1)
    if not use_ddp:
        setup_for_distributed(True)
        if preferred_device == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            dev = preferred_device
        return False, dev, 0, 1

    if preferred_device == 'cpu':
        raise ValueError('DDP FT scaffold requires CUDA. Set runtime.device=cuda/auto and launch with torchrun.')
    local_rank = int(os.environ.get('LOCAL_RANK', str(getattr(runtime, 'local_rank', 0))))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=str(getattr(runtime, 'dist_backend', 'nccl')),
        init_method=str(getattr(runtime, 'dist_url', 'env://')),
    )
    rank = get_rank()
    world_size = get_world_size()
    setup_for_distributed(rank == 0)
    dist.barrier()
    return True, f'cuda:{local_rank}', rank, world_size


def ddp_cleanup():
    if ddp_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
