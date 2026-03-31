from typing import List, Optional
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class DataCfg:
    data_root: str = "/data/user/proj/nfm/data/20260111_Tseng_train"
    pattern: str = "*AR_70_15_15_allplanes.npz"
    exclude_dir_prefix: List[str] = field(default_factory=lambda: ["prev_"])
    sample_batch: int = 256
    batch_size: int = 1
    num_workers: int = 4
    repeat_factor: int = 1
    align_mod: int = 4
    preload: bool = True
    pin_memory: bool = True

@dataclass
class VQModelCfg:
    discretization_window: int = 4
    overlap_window: int = 4
    n_emb: int = 128
    dim_emb: int = 512
    heads: int = 4
    trans_layer_num_enc: int = 4
    trans_layer_num_dec: int = 4
    dropout_ratio: float = 0.2
    decay: float = 0.99
    epsilon: float = 1e-5
    use_gumbel: bool = True
    use_gumbel_hard: bool = True
    temperature_high: float = 2.5
    temperature_low: float = 0.01
    reset_threshold_ratio: float = 0.2
    dead_code_ema_reset_val: float = 1.0
    use_periodic_kmeans_recluster: bool = True
    z_e_buffer_capacity: int = 10000
    min_z_e_for_recluster: int = 1000
    encoder_causal: bool = True
    decoder_causal: bool = True
    lookahead_tokens: int = 0
    max_ar_k: int = 4
    temporal_ema_alpha: float = .9
    trans_layer_num_ar: int = 2
    trans_layer_num_mlm: int = 2
    ar_dropout: float = .1

@dataclass
class OptimCfg:
    lr: float = 5e-5
    weight_decay: float = 5e-6
    weight_decay_no_decay: float = 0.0
    eta_min: float = 1e-6

@dataclass
class TrainerCfg:
    epoch_max: int = 100
    annealing_epoch: int = 80
    epoch_reset: int = 10
    epoch_warmup: int = 1
    recluster_interval_epoch: int = 20
    recluster_end_epoch: int = 80
    evaluation_interval: int = 2
    initial_ema_count_after_recluster: float = 1.

@dataclass
class LossCfg:
    w_embedding: float = 1.0
    w_commitment: float = 0.5
    w_entropy: float = 0.5
    w_fft: float = 1e-9
    w_hfp: float = 1e-9
    extra_loss: str = "pearson" 
    extra_loss_ratio: float = .5
    w_orth: float = 5e-6
    w_latent_tv: float = 5e-2
    w_latent_tv2: float = 5e-4
    w_logit_js: float = 2e-3
    logit_js_tau: float = 1.0
    w_sticky: float = 1e-1
    w_bigram_condent: float = 5e-1
    w_bigram_ce: float = 5e-1
    w_bigram_align: float = 5e-1
    ar_k_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    w_ar_k_ce: float = 5e-1
    w_ar_align: float = 5e-1
    w_cpc: float = 5e-1
    cpc_tau: float = 0.1
    w_mlm_ce: float = 0.0
    mlm_mask_prob: float = 0.15
    condent_bits: bool = True
    condent_norm: bool = True
    ce_label_smoothing: float = 0.05

@dataclass
class PathsCfg:
    output_root: str = "/data/user/proj/nfm/runs/vq"
    ckpt_dir: str = "${hydra:runtime.output_dir}/checkpoints"

@dataclass
class LoggingCfg:
    use_wandb: bool = False
    plot_dir: Optional[str] = None
    plot_every: int = 1

@dataclass
class RuntimeCfg:
    device: str = "cuda"
    tf32: bool = True
    matmul_precision: str = "high"
    compile: bool = True
    seed: int = 0

@dataclass
class VQTrainCfg:
    remark: str = "vq_default"
    data: DataCfg = field(default_factory=DataCfg)
    model: VQModelCfg = field(default_factory=VQModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="vq_train", node=VQTrainCfg)
