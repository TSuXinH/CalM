from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import VQForCalcium, TrainerVQ, PearsonCorrelationLoss
from util import set_seed
from dataset import build_multisession_loaders
from .nq_vq_configs_train_Tseng import VQTrainCfg, register_configs
register_configs()

@hydra.main(version_base="1.3", config_path="../conf/nq", config_name="vq_train_Tseng")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.runtime.seed))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.runtime.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.runtime.tf32)
    try: torch.set_float32_matmul_precision(str(cfg.runtime.matmul_precision))
    except Exception: pass
    device = str(cfg.runtime.device)
    train_loader, val_loader, npz_list = build_multisession_loaders(
        data_root=str(cfg.data.data_root),
        pattern=str(cfg.data.pattern),
        exclude_dir_prefix=tuple(cfg.data.exclude_dir_prefix),
        sample_batch=int(cfg.data.sample_batch),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        repeat_factor=int(cfg.data.repeat_factor),
        align_mod=int(cfg.data.align_mod),
        preload=bool(cfg.data.preload),
        pin_memory=bool(cfg.data.pin_memory),
    )
    print("Loaded npz files:", len(npz_list))
    m = cfg.model
    model = VQForCalcium(
        int(m.discretization_window),
        int(m.overlap_window),
        int(m.n_emb),
        int(m.dim_emb),
        int(m.heads),
        int(m.trans_layer_num_enc),
        int(m.trans_layer_num_dec),
        dropout_ratio=float(m.dropout_ratio),
        decay=float(m.decay),
        epsilon=float(m.epsilon),
        use_gumbel=bool(m.use_gumbel),
        use_gumbel_hard=bool(m.use_gumbel_hard),
        temperature=float(m.temperature_high),
        temperature_entropy=float(m.temperature_high),
        reset_threshold_ratio=float(m.reset_threshold_ratio),
        dead_code_ema_reset_val=float(m.dead_code_ema_reset_val),
        use_periodic_kmeans_recluster=bool(m.use_periodic_kmeans_recluster),
        z_e_buffer_capacity=int(m.z_e_buffer_capacity),
        min_z_e_for_recluster=int(m.min_z_e_for_recluster),
        temporal_ema_alpha=float(m.temporal_ema_alpha),
        trans_layer_num_ar=int(m.trans_layer_num_ar),
        trans_layer_num_mlm=int(m.trans_layer_num_mlm),
        max_ar_k=int(m.max_ar_k),
        ar_dropout=float(m.ar_dropout),
        lookahead_tokens=int(m.lookahead_tokens),
        encoder_causal=bool(m.encoder_causal),
        decoder_causal=bool(m.decoder_causal),
    )
    model = model.to(device)
    if bool(cfg.runtime.compile): model = torch.compile(model)
    criterion = nn.MSELoss()
    extra_loss = None
    if str(cfg.loss.extra_loss).lower() == "pearson": extra_loss = PearsonCorrelationLoss()
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or ("codebook" in n) or ("bigram_logits" in n): no_decay_params.append(p)
        else: decay_params.append(p)
    optimizer = optim.AdamW([
            {"params": decay_params, "weight_decay": float(cfg.optim.weight_decay)},
            {"params": no_decay_params, "weight_decay": float(cfg.optim.weight_decay_no_decay)},
        ], lr=float(cfg.optim.lr),
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.trainer.epoch_max), eta_min=float(cfg.optim.eta_min),)
    ckpt_dir = Path(OmegaConf.to_container(cfg.paths, resolve=True)["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    l = cfg.loss
    t = cfg.trainer
    trainer = TrainerVQ(
        model,
        int(t.epoch_max),
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        float(l.w_embedding),
        float(l.w_commitment),
        str(ckpt_dir),
        w_entropy=float(l.w_entropy),
        use_gumbel=bool(m.use_gumbel),
        temperature_start=float(m.temperature_high),
        temperature_end=float(m.temperature_low),
        annealing_epoch=int(t.annealing_epoch),
        epoch_reset=int(t.epoch_reset),
        epoch_warmup=int(t.epoch_warmup),
        recluster_interval_epoch=int(t.recluster_interval_epoch),
        recluster_end_epoch=int(t.recluster_end_epoch),
        initial_ema_count_after_recluster=float(t.initial_ema_count_after_recluster),
        remark=str(cfg.remark),
        evaluation_interval=int(t.evaluation_interval),
        w_hfp=float(l.w_hfp),
        scheduler=scheduler,
        extra_loss=extra_loss,
        extra_loss_ratio=float(l.extra_loss_ratio),
        w_orth=float(l.w_orth),
        w_fft=float(l.w_fft),
        w_latent_tv=float(l.w_latent_tv),
        w_latent_tv2=float(l.w_latent_tv2),
        w_logit_js=float(l.w_logit_js),
        logit_js_tau=float(l.logit_js_tau),
        w_sticky=float(l.w_sticky),
        w_bigram_condent=float(l.w_bigram_condent),
        w_bigram_ce=float(l.w_bigram_ce),
        w_bigram_align=float(l.w_bigram_align),
        ar_k_list=tuple(int(x) for x in l.ar_k_list),
        w_ar_k_ce=float(l.w_ar_k_ce),
        w_ar_align=float(l.w_ar_align),
        w_cpc=float(l.w_cpc),
        cpc_tau=float(l.cpc_tau),
        w_mlm_ce=float(l.w_mlm_ce),
        mlm_mask_prob=float(l.mlm_mask_prob),
        condent_bits=bool(l.condent_bits),
        condent_norm=bool(l.condent_norm),
        ce_label_smoothing=float(l.ce_label_smoothing),
        plot_dir=cfg.logging.plot_dir,
        plot_every=int(cfg.logging.plot_every),
    )
    trainer.train()

if __name__ == "__main__":
    main()
