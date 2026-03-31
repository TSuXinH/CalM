from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class FTDataConfig:
    registry_json: str = '/data/user/proj/nfm/ckpt/pretrain_AR/20260122_ddp_ft/registry.json'
    heldin_root: Optional[str] = '/data/user/proj/nfm/data/20260112_Tseng_train_token'
    heldout_root: Optional[str] = None
    token_key: str = 'token'
    beh_key: str = 'behavior'
    vocab: int = 128
    beh_channels: int = 3
    beh_up: int = 4
    batch_size: int = 8
    num_workers: int = 8


@dataclass
class FTBackboneConfig:
    ar_ckpt: str = '/data/user/proj/nfm/ckpt/pretrain_AR/20260122_ddp_ft/axial_ar_ep160.pth'
    prefer_ema_weights: bool = True
    compile: bool = False
    compile_dynamic: bool = True
    init_head_ckpt: Optional[str] = None
    init_head_registry_json: Optional[str] = None


@dataclass
class FTHeadConfig:
    name: str = 'lowrank_uv'
    rank: int = 16
    dropout: float = 0.5
    smooth_kernel: int = 9
    conv_init_identity: bool = True
    conv_bias: bool = True


@dataclass
class FTTrainBaseConfig:
    enabled: bool = False
    epochs: int = 200
    lr: float = 0.0036
    weight_decay: float = 1e-2
    warmup_frac: float = 0.1
    eval_every_epochs: int = 5
    do_test_during_train: bool = True


@dataclass
class FTTrainHeldoutConfig:
    enabled: bool = False
    epochs: int = 100
    lr: float = 0.0072
    weight_decay: float = 1e-3
    warmup_frac: float = 0.1
    eval_every_epochs: int = 5
    do_test_during_train: bool = True
    ft_mode: str = 'all'   # all / newrows


@dataclass
class FTCacheConfig:
    enabled: bool = True
    build: bool = True
    use: bool = True
    force: bool = False
    dir: str = '/data/user/proj/nfm/ckpt/finetuned_decoding/ft_behavior/cache'
    dtype: str = 'fp16'    # fp16 / bf16 / fp32
    max_batches: int = -1
    allow_mismatch: bool = False
    allow_empty_eval: bool = True


@dataclass
class FTEvalConfig:
    # These two are the original script semantics.
    eval_held_in: bool = True
    eval_held_out: bool = False
    # Backward-compatible aliases for older scaffold fields.
    eval_held_in_init: bool = False
    eval_held_out_init: bool = False
    max_eval_batches: int = -1
    print_each_session: bool = False
    print_limit_sessions: int = -1


@dataclass
class FTRuntimeConfig:
    device: str = 'auto'
    seed: int = 0
    tf32: bool = True
    matmul_precision: str = 'high'
    use_bf16: bool = True
    ddp: bool = True
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    local_rank: int = 0


@dataclass
class FTPathsConfig:
    output_root: str = '/data/user/proj/nfm/runs/ft'
    save_dir: str = '${hydra:runtime.output_dir}'


@dataclass
class FTTaskConfig:
    remark: str = 'ft_behavior_tseng'
    # cache_only / train_base / finetune_heldout / eval_only / full_pipeline
    mode: str = 'full_pipeline'
    run_name: str = 'heldout_ft'
    data: FTDataConfig = field(default_factory=FTDataConfig)
    backbone: FTBackboneConfig = field(default_factory=FTBackboneConfig)
    head: FTHeadConfig = field(default_factory=FTHeadConfig)
    train_base: FTTrainBaseConfig = field(default_factory=FTTrainBaseConfig)
    train_heldout: FTTrainHeldoutConfig = field(default_factory=FTTrainHeldoutConfig)
    cache: FTCacheConfig = field(default_factory=FTCacheConfig)
    eval: FTEvalConfig = field(default_factory=FTEvalConfig)
    runtime: FTRuntimeConfig = field(default_factory=FTRuntimeConfig)
    paths: FTPathsConfig = field(default_factory=FTPathsConfig)


def register_ft_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name='ft_behavior', node=FTTaskConfig)
