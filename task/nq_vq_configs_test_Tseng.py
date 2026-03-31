from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from hydra.core.config_store import ConfigStore
from task.nq_vq_configs_train_Tseng import VQTrainCfg, register_configs

@dataclass
class TokenizeCfg:
    sub: str = "sub-3_processed"
    date: str = "20260115"
    epoch: int = 8
    keywords: str = "multi-plane-imaging_behavior+ophys_AR_70_15_15_allplanes"
    glob_pattern: str = "**/*{keywords}*"
    out_root_base: str = "/data/user/proj/nfm/data/20260115_Tseng_train_token"
    ckpt_path: str = "/data/user/proj/nfm/ckpt/Tseng/VQ_multi_animal/.../CurE${tokenize.epoch}.pth"
    neuron_chunk: int = 512
    use_amp: bool = False
    align_mod: Optional[int] = 4
    resume_skip_valid: bool = True
    remove_invalid: bool = True
    force_gumbel_off: bool = True
    compile: bool = True
    compile_dynamic: bool = True
    compile_mode: str = "reduce-overhead"

@dataclass
class VQTokenizeCfg(VQTrainCfg):
    tokenize: TokenizeCfg = field(default_factory=TokenizeCfg)

def register_tokenize_configs():
    cs = ConfigStore.instance()
    cs.store(name="vq_test", node=VQTokenizeCfg)
