from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DataDT:
    data_root: str = "/data/user/proj/nfm/data/20260115_Tseng_train_token"
    heldout_root: Optional[str] = "/data/user/proj/nfm/data/20260117_heldout_token_subset_l23"
    registry_json: Optional[str] = "/data/user/proj/nfm/ckpt/pretrain_AR/20260119_causal/registry.json"
    registry_json_heldout: Optional[str] = "/data/user/proj/nfm/ckpt/pretrain_AR/20260119_causal/registry_heldout.json"
    pattern: str = "*.npz"
    exclude_dir_prefix: List[str] = field(default_factory=lambda: ["prev_"])
    preload: bool = True
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    token_key: str = "token"
    vocab: int = 128
    n_sub: int = 512
    batch_size: int = 4
    eval_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class VQCfg:
    vq_state: str = ""
    vq_disc_win: int = 4
    vq_overlap: int = 4
    vq_n_emb: int = 128
    vq_dim_emb: int = 512
    vq_heads: int = 4
    vq_enc_layers: int = 4
    vq_dec_layers: int = 4
    vq_encoder_causal: bool = True
    vq_decoder_causal: bool = False
    lookahead_tokens: int = 0


@dataclass
class ModelDT:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.15
    emb_dropout: float = 0.15
    attn_dropout: float = 0.15
    train_window_T: int = 0
    eval_use_full_trial: bool = True
    use_abs_time_emb: bool = False
    max_tokens_per_trial: int = 0  # kept for CLI parity; original middle script declares it but does not consume it.


@dataclass
class OptimDT:
    lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_ratio: float = 0.06
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-3


@dataclass
class TrainDT:
    mode: str = "train"  # [train, eval_test, finetune_heldout]
    epochs: int = 160
    init_ckpt: Optional[str] = None
    eval_ckpt: Optional[str] = None
    eval_target: str = "base"  # [base, heldout, both]
    eval_use_ema_weights: bool = True
    heldout_epochs: int = 160
    heldout_lr: Optional[float] = None
    heldout_train_mode: str = "embed_only"  # [embed_only, full]
    heldout_init_ckpt: Optional[str] = None
    heldout_init_use_ema_weights: bool = False
    gen_every: int = 10
    gen_prefix: int = 10
    decode_temp: float = 1.0
    decoding_beam: int = 1
    beam_lambda_emb_smooth: float = 0.0
    post_smooth_win: int = 0
    beam_expand_neurons: int = 64
    beam_per_neuron_topk: int = 32
    ckpt_every: int = 10
    eval_detail_every: int = 10
    compile: bool = True
    compile_dynamic: bool = True
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_eval_only: bool = True
    save_ema_weights: bool = True
    use_mtp: bool = False
    mtp_horizons: int = 5
    mtp_hidden: int = 512
    mtp_weight: str = "geom"
    mtp_geom_gamma: float = 0.7
    mtp_warmup_epochs: int = 0
    mtp_lambda: float = 1.0
    use_cf_weights: bool = False
    use_ngram_kd: bool = False
    ngram_order: int = 2
    ngram_alpha: float = 0.2
    ngram_laplace: float = 0.5
    use_bigram_rescore: bool = False
    bigram_alpha: float = 0.0
    use_tok_neighbor_noise: bool = True
    tok_noise_p: float = 0.1
    neighbor_k: int = 12
    use_scheduled_sampling: bool = True
    ss_prob: float = 0.6
    ss_block_len: int = 6
    train_mode: str = "full"  # [full, embed_only]


@dataclass
class RuntimeDT:
    device: str = "auto"  # [auto, cuda, cpu]
    seed: int = 1337
    tf32: bool = True
    matmul_precision: str = "high"
    use_bf16: bool = True
    ddp_backend: str = "nccl"
    deterministic: bool = False
    registry_per_rank: bool = True
    sampler_pad: bool = True


@dataclass
class PathsDT:
    output_root: str = "/data/user/proj/nfm/runs/dt"
    ckpt_dir: str = "${hydra:runtime.output_dir}/checkpoints"
    gen_outdir: str = "${hydra:runtime.output_dir}/gen"


@dataclass
class DTTrainCfg:
    remark: str = "dt_tseng"
    data: DataDT = field(default_factory=DataDT)
    vq: VQCfg = field(default_factory=VQCfg)
    model: ModelDT = field(default_factory=ModelDT)
    optim: OptimDT = field(default_factory=OptimDT)
    train: TrainDT = field(default_factory=TrainDT)
    runtime: RuntimeDT = field(default_factory=RuntimeDT)
    paths: PathsDT = field(default_factory=PathsDT)


def register_dt_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="dt_train", node=DTTrainCfg)
