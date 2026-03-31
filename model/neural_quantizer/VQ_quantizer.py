from .nq_layers import StandardTransformerBlockWithRoPE
from .nq_utility import cal_correlation, compute_ece, brier_score, entropy_rate_from_logits, average_run_length, mi_lag_k, effective_k_fraction, trigram_cond_entropy_bits_hard
import os
import math
import torch
import collections
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from einops import rearrange

class VQForCalcium(nn.Module):
    def __init__(
        self,
        discretization_window,
        overlap_window,
        n_emb,
        dim_emb,
        heads,
        trans_layer_num_enc,
        trans_layer_num_dec,
        dropout_ratio=.2,
        decay=0.99,
        epsilon=1e-5,
        use_gumbel=False,
        use_gumbel_hard=False,
        temperature=1.,
        temperature_entropy=1.,
        reset_threshold_ratio=1e-3,
        dead_code_ema_reset_val=1.,
        use_periodic_kmeans_recluster=True,
        z_e_buffer_capacity=100000,
        min_z_e_for_recluster=1000,
        use_flash=False,
        temporal_ema_alpha: float = 0.9,
        trans_layer_num_ar: int = 2,
        trans_layer_num_mlm: int = 2,
        max_ar_k: int = 3,
        ar_dropout: float = 0.1,
        encoder_causal=False,         
        decoder_causal: bool = False,
        lookahead_tokens: float = 0, 
    ):
        super().__init__()
        enc_list = [StandardTransformerBlockWithRoPE(dim_emb, dim_emb, heads, 4 * dim_emb, dropout_ratio) for _ in range(trans_layer_num_enc)]
        dec_list = [StandardTransformerBlockWithRoPE(dim_emb, dim_emb, heads, 4 * dim_emb, dropout_ratio) for _ in range(trans_layer_num_dec)]
        self.encoder = nn.ModuleList(enc_list)
        self.decoder = nn.ModuleList(dec_list)
        self.encoder_causal = bool(encoder_causal)
        self.decoder_causal = bool(decoder_causal)
        self.lookahead_tokens = lookahead_tokens
        self.enc_discretization_layer = nn.Sequential(
            nn.Conv1d(1, dim_emb // 2, kernel_size=2, stride=2, padding=0, bias=True),
            nn.GELU(),
            nn.Conv1d(dim_emb // 2, dim_emb, kernel_size=discretization_window // 2, stride=overlap_window // 2, padding=0, bias=True),
        )
        self.dec_discretization_layer = nn.Sequential(
            nn.ConvTranspose1d(dim_emb, dim_emb // 2, kernel_size=discretization_window // 2, stride=overlap_window // 2, padding=0, output_padding=0, bias=True),
            nn.GELU(),
            nn.ConvTranspose1d(dim_emb // 2, 1, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True),
        )
        self.temporal_ema_alpha = float(temporal_ema_alpha)
        self.codebook = nn.Embedding(n_emb, dim_emb)
        self.codebook.weight.data.uniform_(-1 / n_emb, 1 / n_emb)
        self.bigram_logits = nn.Parameter(torch.zeros(n_emb, n_emb))
        with torch.no_grad(): self.bigram_logits.data += torch.eye(n_emb) * 2.0
        self.trans_layer_num_ar = trans_layer_num_ar
        self.ar_blocks = nn.ModuleList([StandardTransformerBlockWithRoPE(dim_emb, dim_emb, heads, 4 * dim_emb, dropout_ratio) for _ in range(max(0, trans_layer_num_ar))])
        self.max_ar_k = int(max_ar_k)
        self.ar_k_embed = nn.Embedding(self.max_ar_k + 1, dim_emb)
        nn.init.normal_(self.ar_k_embed.weight, std=0.02)
        self.ar_proj = nn.Linear(dim_emb, dim_emb)
        self.ar_act = nn.GELU()
        self.ar_drop = nn.Dropout(ar_dropout)
        self.ar_head = nn.Linear(dim_emb, n_emb)
        self.cpc_q = nn.Linear(dim_emb, dim_emb)
        self.cpc_k = nn.Linear(dim_emb, dim_emb)
        self.trans_layer_num_mlm = trans_layer_num_mlm
        self.mlm_blocks = nn.ModuleList([StandardTransformerBlockWithRoPE(dim_emb, dim_emb, heads, 4 * dim_emb, dropout_ratio) for _ in range(max(0, trans_layer_num_mlm))])
        self.mlm_head = nn.Linear(dim_emb, n_emb)
        self.mlm_mask_embed = nn.Parameter(torch.zeros(dim_emb))
        nn.init.normal_(self.mlm_mask_embed, std=0.02)
        self.n_emb = n_emb
        self.decay = decay
        self.epsilon = epsilon
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.temperature_entropy = temperature_entropy
        self.use_gumbel_hard = use_gumbel_hard
        self.reset_threshold_ratio = reset_threshold_ratio
        self.dead_code_ema_reset_val = dead_code_ema_reset_val
        self.tag = 'VQ_' + ('Uni_' if self.encoder_causal else 'Bi_') + \
            f'Lookahead{self.lookahead_tokens}_' + \
            f'DiscWin{discretization_window}_OverlapWin{overlap_window}_NEmb{n_emb}_DimEmb{dim_emb}' + \
            f'_Heads{heads}_EncLayer{trans_layer_num_enc}_DecLayer{trans_layer_num_dec}' + \
            (f'_AR{trans_layer_num_ar}' if trans_layer_num_ar > 0 else '') + \
            (f'_MLM{trans_layer_num_mlm}' if trans_layer_num_mlm > 0 else '') + \
            ('_Flash' if use_flash else '') + ('_Gumbel' if use_gumbel else '') + \
            ('_Recluster' if use_periodic_kmeans_recluster else '') + \
            (f'_Reset{reset_threshold_ratio}' if reset_threshold_ratio > 0 else '') + \
            ('_DecCausal' if self.decoder_causal else '')
        print('TAG: ', self.tag)
        self.register_buffer('ema_cluster_size', torch.zeros(n_emb))
        self.register_buffer('ema_dw', self.codebook.weight.data.clone())
        self.use_periodic_kmeans_recluster = use_periodic_kmeans_recluster
        if self.use_periodic_kmeans_recluster:
            self.z_e_buffer_capacity = z_e_buffer_capacity
            self.min_z_e_for_recluster = min_z_e_for_recluster
            self.z_e_buffer = collections.deque(maxlen=z_e_buffer_capacity)

    def get_tag(self):
        return self.tag

    def _ema1d(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha <= 0: return x
        y_prev = x[:, :1, :]
        outs = [y_prev]
        for tt in range(1, x.size(1)):
            y_prev = alpha * x[:, tt:tt+  1, :] + (1 - alpha) * y_prev
            outs.append(y_prev)
        return torch.cat(outs, dim=1)

    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=self.lookahead_tokens + 1)
        return mask

    @staticmethod
    def strict_causal_mask(T, device):
        return torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

    def module_forward(self, x, mask=None, enc=True):
        module = self.encoder if enc else self.decoder
        T = x.size(1)
        for li, coder in enumerate(module):
            if enc and self.encoder_causal: m = self.generate_causal_mask(T).to(x.device) if li == 0 else self.strict_causal_mask(T, x.device)
            else: m = mask
            x = coder(x, mask=m)
        return x

    def ar_context(self, z_e):
        if self.trans_layer_num_ar <= 0: return z_e
        T = z_e.size(1)
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1).to(z_e.device)
        h = z_e
        for blk in self.ar_blocks: h = blk(h, mask=mask)
        return h

    def mlm_context(self, z_e_masked):
        if self.trans_layer_num_mlm <= 0: return z_e_masked
        h = z_e_masked
        for blk in self.mlm_blocks: h = blk(h, mask=None)
        return h

    def forward(self, x, if_reset=False, if_warmup=False, if_training=True):
        assert len(x.shape) == 3, 'Shape of x should be [b, c, t].'  # x: [B,1,T]
        discretized = self.enc_discretization_layer(x)      # [B, C, T']
        b, c, t = discretized.shape
        z_e = self.module_forward(rearrange(discretized, 'b c t -> b t c'), None, enc=True)  # [B,T',C]
        if self.temporal_ema_alpha > 0: z_e = self._ema1d(z_e, self.temporal_ema_alpha)
        z_e_broadcast = z_e.reshape(b, t, 1, c)
        codebook = self.codebook.weight
        k, _ = codebook.shape
        codebook_broadcast = codebook.reshape(1, 1, k, c)
        distance = torch.sum((z_e_broadcast - codebook_broadcast) ** 2, -1)  # [B,T',K]
        nearest_neighbor = torch.argmin(distance, 2)                          # [B,T']
        gumbel_probs_soft_for_entropy = None
        if if_warmup:
            z_q = z_e
            dec_in = z_q
            indices_for_loss = None
        else:
            if self.use_gumbel and if_training:
                logits_distance = -distance.reshape(-1, k) / math.sqrt(c)
                gumbel_probs = F.gumbel_softmax(logits_distance, tau=self.temperature, hard=self.use_gumbel_hard, dim=-1)
                z_q = torch.matmul(gumbel_probs, self.codebook.weight).reshape_as(z_e)
                ema_indices = torch.argmax(gumbel_probs, dim=-1)
                gumbel_probs_soft_for_entropy = F.gumbel_softmax(logits_distance, tau=self.temperature_entropy, hard=False, dim=-1)
                indices_for_loss = ema_indices.view(b, t)
            else:
                z_q = self.codebook(nearest_neighbor)
                indices_for_loss = nearest_neighbor
            dec_in = z_e + (z_q - z_e).detach()
        if if_training and not if_warmup:
            if self.use_periodic_kmeans_recluster: self.z_e_buffer.extend(z_e.reshape(-1, z_e.shape[-1]).detach().cpu().numpy())
            with torch.no_grad():
                ema_indices = indices_for_loss.reshape(-1)
                encodings_one_hot = F.one_hot(ema_indices, self.n_emb).type(z_e.dtype)
                self.ema_cluster_size.mul_(self.decay).add_(torch.sum(encodings_one_hot, dim=0), alpha=(1 - self.decay))
                n_sum = torch.sum(self.ema_cluster_size)
                cluster_size_smooth = (self.ema_cluster_size + self.epsilon) / (n_sum + self.n_emb * self.epsilon) * n_sum
                cluster_size_smooth = cluster_size_smooth.clamp_min(1e-6)
                dw = torch.matmul(encodings_one_hot.t(), z_e.reshape(-1, c))
                self.ema_dw = self.ema_dw * self.decay + (1 - self.decay) * dw
                self.codebook.weight.data.copy_(self.ema_dw / cluster_size_smooth.unsqueeze(1))
                if if_reset:
                    print('\nStart reset.')
                    active_threshold_for_avg = self.epsilon * 10
                    meaningfully_active_counts = self.ema_cluster_size[self.ema_cluster_size > active_threshold_for_avg]
                    avg_cluster_size = torch.mean(meaningfully_active_counts) if len(meaningfully_active_counts) > 0 else torch.mean(self.ema_cluster_size)
                    dead_code_threshold = torch.max(avg_cluster_size * self.reset_threshold_ratio, torch.tensor(self.epsilon, device=avg_cluster_size.device))
                    dead_indices = torch.where(self.ema_cluster_size < dead_code_threshold)[0]
                    print('Dead code threshold: {}, current minimal cluster occupation: {}'.format(dead_code_threshold, torch.min(self.ema_cluster_size)))
                    if len(dead_indices) > 0:
                        z_e_flatten = z_e.reshape(-1, z_e.shape[-1]).detach()
                        num_dead_to_reset = len(dead_indices)
                        if z_e_flatten.shape[0] >= num_dead_to_reset: rand_idx = torch.randperm(z_e_flatten.shape[0], device=z_e_flatten.device)[:num_dead_to_reset]
                        else: rand_idx = torch.randint(0, z_e_flatten.shape[0], (num_dead_to_reset,), device=z_e_flatten.device)
                        replacement_vectors = z_e_flatten[rand_idx]
                        self.codebook.weight.data[dead_indices] = replacement_vectors
                        self.ema_cluster_size.data[dead_indices] = self.dead_code_ema_reset_val
                        self.ema_dw.data[dead_indices] = replacement_vectors * self.dead_code_ema_reset_val
                    else: print('No need to reset here.\n')
        dec_mask = self.generate_causal_mask(z_e.size(1)).to(z_e.device) if self.decoder_causal else None
        dec_h = self.module_forward(dec_in, dec_mask, enc=False)
        dec_h = rearrange(dec_h, 'b t c -> b c t')
        decoded_output = self.dec_discretization_layer(dec_h)
        if if_warmup: return decoded_output, z_e, z_e, None, None
        return decoded_output, z_e, z_q, gumbel_probs_soft_for_entropy, indices_for_loss

    @torch.no_grad()
    def initialize_codebook_with_kmeans(
        self,
        data_loader_for_init,
        num_batches_for_kmeans=16,
        kmeans_n_init=10,
        kmeans_max_iter=300,
        device=None
    ):
        self.eval()
        if device is None: device = next(self.parameters()).device
        z_e_flatten_list = []
        batches_processed = 0
        for batch_trace in data_loader_for_init:
            batch_trace = batch_trace.to(device)
            discretized = self.enc_discretization_layer(batch_trace)
            z_e_batch = self.module_forward(rearrange(discretized, 'b c t -> b t c'), None, enc=True)
            z_e_flatten_list.append(z_e_batch.reshape(-1, z_e_batch.shape[-1]).cpu().numpy())
            batches_processed += 1
            if batches_processed >= num_batches_for_kmeans: break
        z_e_flatten_arr = np.concatenate(z_e_flatten_list, axis=0)
        actual_n_clusters = min(self.n_emb, z_e_flatten_arr.shape[0])
        if actual_n_clusters < self.n_emb: print(f"KMeans Init Warning: Using {actual_n_clusters} clusters (fewer than n_emb={self.n_emb}).")
        print(f"KMeans Init: Running K-Means with {actual_n_clusters} clusters on {z_e_flatten_arr.shape[0]} z_e samples...")
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init=kmeans_n_init, max_iter=kmeans_max_iter, verbose=0)
        kmeans.fit(z_e_flatten_arr)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(dtype=self.codebook.weight.dtype, device=self.codebook.weight.device)
        if actual_n_clusters < self.n_emb:
            self.codebook.weight.data[:actual_n_clusters] = centroids
            self.codebook.weight.data[actual_n_clusters:] = torch.rand(self.n_emb - actual_n_clusters, z_e_flatten_arr.shape[-1], device=device) * (2 / self.n_emb) - (1 / self.n_emb)
        else: self.codebook.weight.data = centroids
        self.ema_dw.data.copy_(self.codebook.weight.data)
        self.ema_cluster_size.data.fill_(1.0)
        print(f"KMeans Init: Codebook initialized. Centroids shape: {centroids.shape}")
        self.train()

    @torch.no_grad()
    def perform_kmeans_recluster_and_reset_ema(
        self,
        kmeans_n_init=10,
        kmeans_max_iter=100,
        initial_ema_count=1.,
    ):
        self.eval()
        print()
        if not hasattr(self, 'z_e_buffer') or len(self.z_e_buffer) < self.min_z_e_for_recluster:
            print('Buffer samples {} are not enough. '.format(0 if not hasattr(self, 'z_e_buffer') else len(self.z_e_buffer)))
            self.train(); print(); return False
        z_e_samples_np = np.array(list(self.z_e_buffer))
        actual_n_clusters = min(self.n_emb, z_e_samples_np.shape[0])
        print(f"KMeans Recluster: Running K-Means with {actual_n_clusters} clusters on {z_e_samples_np.shape[0]} buffered z_e samples...")
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init=kmeans_n_init, max_iter=kmeans_max_iter, verbose=0)
        kmeans.fit(z_e_samples_np)
        new_centroids = torch.from_numpy(kmeans.cluster_centers_).to(dtype=self.codebook.weight.dtype, device=self.codebook.weight.device)
        self.codebook.weight.data[:actual_n_clusters] = new_centroids
        if actual_n_clusters < self.n_emb: self.codebook.weight.data[actual_n_clusters:].uniform_(-1.0 / self.n_emb, 1.0 / self.n_emb)
        self.ema_dw.data.copy_(self.codebook.weight.data * initial_ema_count)
        self.ema_cluster_size.data.fill_(initial_ema_count)
        print(); self.train(); return True

    @torch.no_grad()
    def encoding(self, x):
        x = x.unsqueeze(0).unsqueeze(0) if x.dim() == 1 else x.unsqueeze(0) if x.dim() == 2 else x
        x = x.to(self.codebook.weight.device)
        discretized = self.enc_discretization_layer(x)
        b, c, t = discretized.shape
        z_e = self.module_forward(rearrange(discretized, 'b c t -> b t c'), None, enc=True)
        if self.temporal_ema_alpha > 0: z_e = self._ema1d(z_e, self.temporal_ema_alpha)
        z_e_broadcast = z_e.reshape(b, t, 1, c)
        codebook = self.codebook.weight
        k, _ = codebook.shape
        codebook_broadcast = codebook.reshape(1, 1, k, c)
        distance = torch.sum((z_e_broadcast - codebook_broadcast) ** 2, -1)
        nearest_neighbor = torch.argmin(distance, 2)
        z_q = self.codebook(nearest_neighbor)
        return z_q, nearest_neighbor

    @torch.no_grad()
    def decoding(self, z_q):
        b, t, c = z_q.shape
        dec_mask = self.generate_causal_mask(t).to(z_q.device) if self.decoder_causal else None
        dec_h = self.module_forward(z_q, dec_mask, enc=False)
        dec_h = rearrange(dec_h, 'b t c -> b c t')
        out = self.dec_discretization_layer(dec_h)
        return out

    @torch.no_grad()
    def decoding_with_token(self, token):
        z_q = self.codebook(token)
        return self.decoding(z_q)

    def calculate_entropy_from_gumbel_probs(self, gumbel_probs_soft, log_smoothing_eps=1e-12):
        if gumbel_probs_soft is None: return torch.tensor(0.0, device=self.codebook.weight.device)
        avg_p_k_batch = torch.mean(gumbel_probs_soft, dim=0).clamp_min(log_smoothing_eps)
        entropy_bits = -(avg_p_k_batch * torch.log2(avg_p_k_batch)).sum()
        return entropy_bits

    def set_temperature(self, temperature, temperature_entropy):
        self.temperature = temperature
        self.temperature_entropy = temperature_entropy

    def clear_buffer(self):
        if hasattr(self, 'z_e_buffer'): self.z_e_buffer.clear()
        else: print('No z_e buffer is initialized.')

class TrainerVQ:
    def __init__(
        self,
        VQ_model,
        epoch_max,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        device,
        w_embedding,
        w_commitment,
        checkpoint_directory,
        w_entropy=0.,
        use_gumbel=False,
        temperature_start=1.,
        temperature_end=.1,
        annealing_epoch=60,
        epoch_reset=5,
        epoch_warmup=5,
        recluster_interval_epoch=10,
        recluster_end_epoch=80,
        initial_ema_count_after_recluster=1.0,
        remark=None,
        evaluation_interval=5,
        w_hfp=1e-9,
        scheduler=None,
        extra_loss=None,
        extra_loss_ratio=.5,
        w_orth=5e-6,
        w_fft=1e-6,
        w_latent_tv: float = 1e-2,
        w_latent_tv2: float = 0,
        w_logit_js: float = 0,
        logit_js_tau: float = 1.0,
        w_sticky: float = 5e-3,
        w_bigram_condent: float = 1e-1,
        w_bigram_ce: float = 1e-1,
        w_bigram_align: float = 1e-1,
        ar_k_list=(1, 2, 3),
        w_ar_k_ce: float = 5e-2,
        w_ar_align: float = 5e-2,
        w_cpc: float = 1e-2,
        cpc_tau: float = 0.07,
        w_mlm_ce: float = 5e-2,
        mlm_mask_prob: float = 0.15,
        condent_bits: bool = True,
        condent_norm: bool = True,
        ce_label_smoothing: float = 0.05,
        plot_dir: str = './',
        plot_every: int = 1,
    ):
        self.model = VQ_model
        self.epoch_max = epoch_max
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.w_embedding = w_embedding
        self.w_commitment = w_commitment
        self.evaluation_interval = evaluation_interval
        remark = '_Remark_' + remark if remark else ''
        self.ckpt_name = self.model.get_tag() + '_E{}'.format(self.epoch_max) + \
            '_WEmb{:.2f}'.format(self.w_embedding) + '_WCommit{:.2f}'.format(self.w_commitment) + \
            '_Ent{}'.format(w_entropy) + '_hfp{:.2f}e-4'.format(w_hfp * 10 ** 4) + \
            '_FFT{:.2f}e-4'.format(w_fft * 10 ** 4) + ('_ExtLoss' if extra_loss else '') + remark
        print('CKPT NAME: ', self.ckpt_name)
        self.ckpt_path = os.path.join(checkpoint_directory, self.ckpt_name)
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        self.scheduler = scheduler
        self.extra_loss = extra_loss
        self.extra_loss_ratio = extra_loss_ratio
        self.w_fft = w_fft
        self.w_orth = w_orth
        self.w_entropy = w_entropy
        self.epoch_reset = epoch_reset
        self.epoch_warmup = epoch_warmup
        self.use_gumbel = use_gumbel
        self.cur_gumbel_temp_entropy = temperature_start
        self.cur_gumbel_temp_z_q = temperature_start
        assert temperature_start > temperature_end, 'Gumbel temperature must anneal from high to low, instead of from {} to {}'.format(temperature_start, temperature_end)
        self.final_gumbel_temp = temperature_end
        self.annealing_epoch = annealing_epoch
        self.gumbel_temp_decay_rate = (temperature_end / temperature_start)**(1.0 / annealing_epoch)
        self.recluster_interval_epoch = recluster_interval_epoch
        self.recluster_end_epoch = recluster_end_epoch
        self.initial_ema_count_after_recluster = initial_ema_count_after_recluster
        self.w_hfp = w_hfp
        self.w_latent_tv = w_latent_tv
        self.w_latent_tv2 = w_latent_tv2
        self.w_logit_js = w_logit_js
        self.logit_js_tau = logit_js_tau
        self.w_sticky = w_sticky
        self.w_bigram_condent = w_bigram_condent
        self.w_bigram_ce = w_bigram_ce
        self.w_bigram_align = w_bigram_align
        self.ar_k_list = list(ar_k_list)
        self.w_ar_k_ce = w_ar_k_ce
        self.w_ar_align = w_ar_align
        self.w_cpc = w_cpc
        self.cpc_tau = cpc_tau
        self.w_mlm_ce = w_mlm_ce
        self.mlm_mask_prob = mlm_mask_prob
        self.condent_bits = condent_bits
        self.condent_norm = condent_norm
        self.ce_label_smoothing = ce_label_smoothing
        self.log_temporal_metrics = True  # set False for speed
        self.plot_dir = plot_dir
        self.plot_every = max(1, int(plot_every))
        self.history = {'epoch': [], 'PPL1': [], 'PPL2': [], 'PPL3': [], 'AR1@1': [], 'AR2@1': [], 'AR3@1': []}
        if self.plot_dir: os.makedirs(self.plot_dir, exist_ok=True)

    def _freeze_temporal_heads(self, requires_grad: bool):
        self.model.bigram_logits.requires_grad_(requires_grad)
        self.model.mlm_mask_embed.requires_grad_(requires_grad)
        for m in [self.model.ar_head, self.model.ar_proj, self.model.cpc_q, self.model.cpc_k, self.model.mlm_head]:
            for p in m.parameters(): p.requires_grad_(requires_grad)
        for p in self.model.ar_k_embed.parameters(): p.requires_grad_(requires_grad)
        for blk in list(getattr(self.model, 'ar_blocks', [])) + list(getattr(self.model, 'mlm_blocks', [])):
            for p in blk.parameters(): p.requires_grad_(requires_grad)

    def process_warmup(self, epoch):
        self.model.train()
        self._freeze_temporal_heads(False)
        loader = self.train_loader
        list_step_loss_recon, list_step_loss_total, list_step_corr = [], [], []
        for _, batch_trace in tqdm(enumerate(loader)):
            batch_trace = batch_trace.to(self.device)
            recon_trace, _, _, _, _ = self.model(batch_trace, if_reset=False, if_warmup=True, if_training=True,)
            loss_recon = self.criterion(recon_trace, batch_trace)
            if self.extra_loss: loss_recon += self.extra_loss_ratio * self.extra_loss(recon_trace, batch_trace)
            loss_total = loss_recon
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            list_step_loss_recon.append(loss_recon.item())
            list_step_loss_total.append(loss_total.item())
            list_step_corr.append(cal_correlation(batch_trace.squeeze(1).detach(), recon_trace.squeeze(1).detach()).item())
        self._freeze_temporal_heads(True)
        epoch_loss_recon = torch.mean(torch.tensor(list_step_loss_recon))
        epoch_loss_total = torch.mean(torch.tensor(list_step_loss_total))
        epoch_corr = torch.mean(torch.tensor(list_step_corr))
        print('Warmup epoch {}: '.format(epoch+1) +
              'loss recon: {:.4f} | '.format(epoch_loss_recon.item()) +
              'loss total: {:.4f} | '.format(epoch_loss_total.item()) +
              'Corr: {:.4f}'.format(epoch_corr.item()))
        print('Current lr: {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
        return epoch_loss_recon, epoch_loss_total, epoch_corr

    def process(self, epoch, if_training=True):
        self.model.train() if if_training else self.model.eval()
        loader = self.train_loader if if_training else self.valid_loader
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(if_training)
        lists = dict(
            recon=[], embedding=[], commitment=[], hfp=[], fft=[],
            emb=[], tv=[], js=[], entropy=[], bigram_condent=[], bigram_ce=[], bigram_align=[],
            total=[], corr=[], H=[], Hc=[], MI=[], sticky=[], PPL1=[], PPL2=[],
            Hrate_model=[], lambda2=[], gap=[],
            MI_lag2=[], AvgRun=[], EffKFrac=[],
            ar_ce_bits_k1=[], ar_ce_bits_k2=[], ar_ce_bits_k3=[],
            ar_top1_k1=[], ar_top1_k2=[], ar_top1_k3=[],
            mlm_bits=[], mlm_top1=[], mlm_top5=[], mlm_ece=[], mlm_brier=[],
            cpc_bits=[], cpc_top1=[], cpc_top5=[], Hc3=[], PPL3=[],
        )
        if self.use_gumbel:
            if epoch < self.annealing_epoch:
                self.cur_gumbel_temp_entropy = max(self.final_gumbel_temp, self.cur_gumbel_temp_entropy * self.gumbel_temp_decay_rate)
                self.cur_gumbel_temp_z_q = max(self.final_gumbel_temp, self.cur_gumbel_temp_z_q * self.gumbel_temp_decay_rate)
            else:
                self.cur_gumbel_temp_entropy = self.final_gumbel_temp
                self.cur_gumbel_temp_z_q = self.final_gumbel_temp
            self.model.set_temperature(self.cur_gumbel_temp_z_q, self.cur_gumbel_temp_entropy)
        for idx, batch_trace in tqdm(enumerate(loader)):
            batch_trace = batch_trace.to(self.device)
            recon_trace, z_e, z_q, gumbel_probs, idx_codes = self.model(
                batch_trace,
                if_reset=((idx == 0) and ((epoch + 1) % self.epoch_reset == 0)),
                if_warmup=False,
                if_training=if_training,
            )
            loss_recon = self.criterion(recon_trace, batch_trace)
            if self.extra_loss: loss_recon += self.extra_loss_ratio * self.extra_loss(recon_trace, batch_trace)
            loss_embedding = self.criterion(z_e.detach(), z_q)
            loss_commitment = self.criterion(z_q.detach(), z_e)
            loss_total = loss_recon + self.w_commitment * loss_commitment + self.w_embedding * loss_embedding
            if self.w_orth != 0:
                cov = self.model.codebook.weight @ self.model.codebook.weight.t()
                iden = torch.eye(cov.shape[0], device=cov.device)
                loss_emb = self.w_orth * (cov - iden).abs().sum()
                loss_total = loss_total + loss_emb
                lists['emb'].append(loss_emb.item())
            if self.use_gumbel and self.w_entropy != 0:
                entropy_bits = self.model.calculate_entropy_from_gumbel_probs(gumbel_probs)
                loss_entropy = -self.w_entropy * entropy_bits
                loss_total = loss_total + loss_entropy
                lists['entropy'].append(loss_entropy.item())
            if self.w_hfp > 0 and recon_trace.size(-1) >= 3:
                diff2 = recon_trace[:, :, 2:] - 2 * recon_trace[:, :, 1:-1] + recon_trace[:, :, :-2]
                loss_smooth = diff2.abs().mean()
                loss_total = loss_total + self.w_hfp * loss_smooth
                lists['hfp'].append(loss_smooth.item())
            if self.w_fft > 0:
                fft_original = torch.fft.fft(batch_trace, dim=-1)
                fft_recon = torch.fft.fft(recon_trace, dim=-1)
                num_points = batch_trace.shape[-1]
                hf_start_index = num_points // 4
                loss_fft_hf = self.criterion(torch.abs(fft_recon[..., hf_start_index:]), torch.abs(fft_original[..., hf_start_index:]).detach())
                loss_total = loss_total + self.w_fft * loss_fft_hf
                lists['fft'].append(loss_fft_hf.item())
            if self.w_latent_tv > 0 and z_e.size(1) >= 2:
                dz = z_e[:, 1:, :] - z_e[:, :-1, :]
                loss_tv_latent = dz.pow(2).mean()
                loss_total = loss_total + self.w_latent_tv * loss_tv_latent
                lists['tv'].append(loss_tv_latent.item())
                if self.w_latent_tv2 > 0 and z_e.size(1) >= 3:
                    ddz = z_e[:, 2:, :] - 2 * z_e[:, 1:-1, :] + z_e[:, :-2, :]
                    loss_tv2 = ddz.pow(2).mean()
                    loss_total = loss_total + self.w_latent_tv2 * loss_tv2
            need_p = (self.w_logit_js > 0) or (self.w_bigram_condent > 0) or (self.w_bigram_align > 0) or self.log_temporal_metrics
            if need_p:
                cb = self.model.codebook.weight
                B, T, C = z_e.shape
                if T >= 2:
                    z = z_e.unsqueeze(2)                  # [B, T, 1, C]
                    cbw = cb.view(1, 1, cb.size(0), C)    # [1, 1, K, C]
                    dist = torch.sum((z - cbw) ** 2, dim=-1)
                    tau = max(1e-6, float(self.logit_js_tau))
                    logits = -dist / tau
                    p = torch.softmax(logits, dim=-1).clamp_min(1e-9)   # [B, T, K]
                else: p = None
            else: p = None
            if (self.w_logit_js > 0 or self.w_sticky > 0) and (p is not None) and (p.size(1) >= 2):
                p1 = p[:, 1:, :]
                p0 = p[:, :-1, :]
                m = (0.5 * (p0 + p1)).clamp_min(1e-9)
                kl0 = (p0 * (p0.log() - m.log())).sum(dim=-1)
                kl1 = (p1 * (p1.log() - m.log())).sum(dim=-1)
                js = 0.5 * (kl0 + kl1)
                loss_js = js.mean()
                stay_prob = (p[:, 1:, :] * p[:, :-1, :]).sum(dim=-1)
                loss_sticky = (1.0 - stay_prob).mean()
                loss_total = loss_total + self.w_sticky * loss_sticky + self.w_logit_js * loss_js
                lists['js'].append(loss_js.item())
            if self.log_temporal_metrics and (p is not None) and (p.size(1) >= 2):
                with torch.no_grad():
                    Ktok = p.size(-1)
                    q = p.mean(dim=(0, 1)).clamp_min(1e-12)                 # [K]
                    H_bits = -(q * torch.log2(q)).sum()
                    p_prev = p[:, :-1, :]
                    p_next = p[:,  1:, :]
                    p_prev_f = p_prev.reshape(-1, Ktok)
                    p_next_f = p_next.reshape(-1, Ktok)
                    joint = (p_prev_f.T @ p_next_f) / p_prev_f.size(0)      # [K, K]
                    marg_prev = joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    cond = (joint / marg_prev).clamp_min(1e-12)
                    Hc_bits = -(joint * torch.log2(cond)).sum()
                    MI_bits = H_bits - Hc_bits
                    if idx_codes is not None and idx_codes.size(1) >= 2: sticky_hard = (idx_codes[:, 1:] == idx_codes[:, :-1]).float().mean()
                    else: sticky_hard = (p[:, 1:, :] * p[:, :-1, :]).sum(dim=-1).mean()
                    PPL1 = torch.exp2(H_bits)
                    PPL2 = torch.exp2(Hc_bits)
                    lists['H'].append(H_bits.item()); lists['Hc'].append(Hc_bits.item()); lists['MI'].append(MI_bits.item())
                    lists['sticky'].append(sticky_hard.item())
                    lists['PPL1'].append(PPL1.item()); lists['PPL2'].append(PPL2.item())
                    lists['EffKFrac'].append(effective_k_fraction(H_bits, q.numel()))
            if (self.w_bigram_condent > 0) and (p is not None) and (p.size(1) >= 2):
                Ktok = p.size(-1)
                p_prev = p[:, :-1, :]
                p_next = p[:,  1:, :]
                p_prev_f = p_prev.reshape(-1, Ktok)
                p_next_f = p_next.reshape(-1, Ktok)
                joint = (p_prev_f.T @ p_next_f) / p_prev_f.size(0)         # [K, K]
                marg_prev = joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
                cond = (joint / marg_prev).clamp_min(1e-12)
                Hc_bits = -(joint * torch.log2(cond)).sum()
                if self.condent_norm: Hc_bits = Hc_bits / torch.log2(torch.tensor(float(Ktok), device=joint.device))
                loss_bigram_condent = - self.w_bigram_condent * Hc_bits
                loss_total = loss_total + loss_bigram_condent
                lists['bigram_condent'].append(loss_bigram_condent.item())
            if (self.w_bigram_ce > 0 or self.w_bigram_align > 0) and (p is not None) and (p.size(1) >= 2) and (idx_codes is not None):
                Ktok = p.size(-1)
                device = p.device
                p_next = p[:, 1:, :]
                prev_idx = idx_codes[:, :-1].reshape(-1)
                next_idx = idx_codes[:,  1:].reshape(-1)
                if self.w_bigram_ce > 0:
                    logP = torch.log_softmax(self.model.bigram_logits, dim=-1)           # [K, K]
                    nll = -logP[prev_idx, next_idx]
                    loss_bigram_ce = self.w_bigram_ce * nll.mean()
                    loss_total = loss_total + loss_bigram_ce
                    lists['bigram_ce'].append(loss_bigram_ce.item())
                if self.w_bigram_align > 0:
                    p_pred = torch.softmax(self.model.bigram_logits[prev_idx], dim=-1)   # [B*(T-1), K]
                    p_next_flat = p_next.reshape(-1, Ktok).detach()
                    loss_bigram_align = (-(p_next_flat * (p_pred.clamp_min(1e-12)).log()).sum(dim=-1).mean())
                    loss_bigram_align = self.w_bigram_align * loss_bigram_align
                    loss_total = loss_total + loss_bigram_align
                    lists['bigram_align'].append(loss_bigram_align.item())
                with torch.no_grad():
                    Hrate_bits, lambda2, gap = entropy_rate_from_logits(self.model.bigram_logits.detach().cpu())
                    lists['Hrate_model'].append(float(Hrate_bits))
                    lists['lambda2'].append(lambda2); lists['gap'].append(gap)
            if z_e.size(1) >= 2 and len(self.ar_k_list) > 0:
                h_ar = self.model.ar_context(z_e) if self.model.trans_layer_num_ar > 0 else z_e
                B, T, C = h_ar.shape
                Ktok = self.model.codebook.num_embeddings
                L = getattr(self.model, "lookahead_tokens", 0)
                for k in self.ar_k_list:
                    if k > self.model.max_ar_k: continue
                    if T <= k: continue
                    kvec = self.model.ar_k_embed.weight[k]  # [C]
                    h_k = h_ar[:, :-k, :] + kvec.view(1, 1, -1)
                    logits_k = self.model.ar_head(self.model.ar_drop(self.model.ar_act(self.model.ar_proj(h_k))))  # [B, T-k, K]
                    target_k = idx_codes[:, k:].reshape(-1)
                    probs_k = torch.softmax(logits_k.reshape(-1, Ktok), dim=-1)
                    ce_nats = F.cross_entropy(
                        logits_k.reshape(-1, Ktok), target_k,
                        reduction='mean',
                        label_smoothing=self.ce_label_smoothing if self.ce_label_smoothing > 0 else 0.0
                    )
                    ce_bits = ce_nats / torch.log(torch.tensor(2.0, device=ce_nats.device))
                    loss_total = loss_total + self.w_ar_k_ce * ce_bits
                    if (k == 1) and (self.w_ar_align > 0) and (p is not None) and (p.size(1) >= 2):
                        p_next_flat = p[:, 1:, :].reshape(-1, Ktok).detach()
                        ar_align_loss = -(p_next_flat * (probs_k.clamp_min(1e-12)).log()).sum(dim=-1).mean()
                        ar_align_bits = ar_align_loss / torch.log(torch.tensor(2.0, device=ar_align_loss.device))
                        loss_total = loss_total + self.w_ar_align * ar_align_bits
                    with torch.no_grad():
                        pred_top1 = probs_k.argmax(dim=-1)
                        top1 = (pred_top1 == target_k).float().mean().item()
                        if k == 1:
                            lists['ar_ce_bits_k1'].append(float(ce_bits.item()))
                            lists['ar_top1_k1'].append(top1)
                        elif k == 2:
                            lists['ar_ce_bits_k2'].append(float(ce_bits.item()))
                            lists['ar_top1_k2'].append(top1)
                        elif k == 3:
                            lists['ar_ce_bits_k3'].append(float(ce_bits.item()))
                            lists['ar_top1_k3'].append(top1)
            if (self.w_cpc > 0) and (z_e.size(1) >= 2):
                zq = F.normalize(self.model.cpc_q(z_e[:, :-1, :]), dim=-1)   # [B, T-1, C]
                zk = F.normalize(self.model.cpc_k(z_e[:,  1:, :]), dim=-1)   # [B, T-1, C]
                N = zq.size(0) * zq.size(1)
                zq_flat = zq.reshape(N, -1)
                zk_flat = zk.reshape(N, -1)
                M_max = 2048
                if N > M_max:
                    idx_sub = torch.randperm(N, device=z_e.device)[:M_max]
                    zq_flat = zq_flat[idx_sub]
                    zk_flat = zk_flat[idx_sub]
                    N = zq_flat.size(0)
                logits_cpc = (zq_flat @ zk_flat.t()) / max(1e-6, self.cpc_tau)  # [N, N]
                labels_cpc = torch.arange(N, device=z_e.device)
                cpc_ce_nats = F.cross_entropy(logits_cpc, labels_cpc, reduction='mean')
                cpc_ce_bits = cpc_ce_nats / torch.log(torch.tensor(2.0, device=z_e.device))
                loss_total = loss_total + self.w_cpc * cpc_ce_bits
                lists['cpc_bits'].append(float(cpc_ce_bits.item()))
                with torch.no_grad():
                    preds = logits_cpc.argmax(dim=1)
                    acc1 = (preds == labels_cpc).float().mean()
                    k = min(5, N)
                    topk_idx = torch.topk(logits_cpc, k=k, dim=1).indices
                    acc5 = (topk_idx == labels_cpc.unsqueeze(1)).any(dim=1).float().mean()
                    lists['cpc_top1'].append(float(acc1.item()))
                    lists['cpc_top5'].append(float(acc5.item()))
            if (self.w_mlm_ce > 0) and (self.model.trans_layer_num_mlm > 0) and (idx_codes is not None):
                B, T = idx_codes.shape
                Ktok = self.model.codebook.num_embeddings
                mask = (torch.rand(B, T, device=z_e.device) < self.mlm_mask_prob)
                if mask.any():
                    z_in = z_e.clone()
                    sel = torch.rand_like(mask.float())
                    m_mask = mask & (sel < 0.8)
                    z_in[m_mask] = self.model.mlm_mask_embed
                    m_rand = mask & (sel >= 0.8) & (sel < 0.9)
                    num_rand = int(m_rand.sum().item())
                    if num_rand > 0:
                        rand_ids = torch.randint(0, Ktok, (num_rand,), device=z_e.device)
                        z_in[m_rand] = self.model.codebook.weight[rand_ids]
                    h_mlm = self.model.mlm_context(z_in)
                    logits_mlm = self.model.mlm_head(h_mlm)       # [B, T, K]
                    targets = idx_codes[mask]
                    if targets.numel() > 0:
                        ce_nats = F.cross_entropy(
                            logits_mlm[mask], targets,
                            reduction='mean',
                            label_smoothing=self.ce_label_smoothing if self.ce_label_smoothing > 0 else 0.0
                        )
                        ce_bits = ce_nats / torch.log(torch.tensor(2.0, device=ce_nats.device))
                        loss_total = loss_total + self.w_mlm_ce * ce_bits
                        lists['mlm_bits'].append(float(ce_bits.item()))
                        with torch.no_grad():
                            probs_mlm = torch.softmax(logits_mlm[mask], dim=-1)
                            pred_top1 = probs_mlm.argmax(dim=-1)
                            top1 = (pred_top1 == targets).float().mean().item()
                            lists['mlm_top1'].append(top1)
                            k5 = min(5, Ktok)
                            topk_idx = torch.topk(probs_mlm, k=k5, dim=-1).indices
                            top5 = (topk_idx == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
                            lists['mlm_top5'].append(top5)
                            ece = compute_ece(probs_mlm, targets, n_bins=15)
                            br = brier_score(probs_mlm, targets, Ktok)
                            lists['mlm_ece'].append(float(ece.item()))
                            lists['mlm_brier'].append(float(br.item()))
            if (idx_codes is not None) and (idx_codes.size(1) >= 3):
                Ktok = self.model.codebook.num_embeddings
                Hc3_bits = trigram_cond_entropy_bits_hard(idx_codes, Ktok)
                lists['Hc3'].append(Hc3_bits)
                lists['PPL3'].append(float(torch.exp2(torch.tensor(Hc3_bits)).item()))
            if idx_codes is not None and idx_codes.size(1) >= 2:
                Ktok = self.model.codebook.num_embeddings
                lists['AvgRun'].append(average_run_length(idx_codes))
                if idx_codes.size(1) >= 3: lists['MI_lag2'].append(mi_lag_k(idx_codes, Ktok, lag=2))
            if if_training:
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
            lists['recon'].append(loss_recon.item())
            lists['total'].append(loss_total.item())
            lists['embedding'].append(loss_embedding.item())
            lists['commitment'].append(loss_commitment.item())
            x_ = batch_trace.squeeze(1).detach()
            y_ = recon_trace.squeeze(1).detach()
            lists['corr'].append(cal_correlation(x_, y_).item())
        if if_training is False: torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, f'CurE{epoch + 1}.pth'))
        if self.scheduler and if_training: self.scheduler.step()
        def M(key, default=0.0):
            arr = lists[key]
            return float(torch.tensor(arr).mean().item()) if len(arr) else default
        msg = ('Epoch {}: '.format(epoch + 1) if if_training else '   Evaluate: ') + \
            f"recon: {M('recon'):.4f} | commit: {M('commitment'):.4f} | " + \
            (f"hfp: {M('hfp'):.4f} | " if len(lists['hfp']) else "") + \
            (f"tv: {M('tv'):.4f} | " if len(lists['tv']) else "") + \
            (f"js: {M('js'):.4f} | " if len(lists['js']) else "") + \
            (f"fft: {M('fft'):.4f} | " if len(lists['fft']) else "") + \
            (f"biCE: {M('bigram_ce'):.4f} | " if len(lists['bigram_ce']) else "") + \
            (f"biAlign: {M('bigram_align'):.4f} | " if len(lists['bigram_align']) else "") + \
            (f"biHcond: {M('bigram_condent'):.4f} | " if len(lists['bigram_condent']) else "") + \
            (f"CPC: {M('cpc_bits'):.3f}b | " if len(lists['cpc_bits']) else "") + \
            (f"MLM: {M('mlm_bits'):.3f}b | " if len(lists['mlm_bits']) else "") + \
            f"total: {M('total'):.4f} | " + \
            (f"H: {M('H'):.2f}b | Hc: {M('Hc'):.2f}b | MI: {M('MI'):.2f}b | " if len(lists['H']) else "") + \
            (f"PPL1: {M('PPL1'):.1f} | PPL2: {M('PPL2'):.1f} | " if len(lists['PPL1']) else "") + \
            (f"Hc3: {M('Hc3'):.2f}b | PPL3: {M('PPL3'):.1f} | " if len(lists['Hc3']) else "") + \
            (f"Sticky: {100*M('sticky'):.2f}% | " if len(lists['sticky']) else "") + \
            (f"Hrate(model): {M('Hrate_model'):.2f}b | λ2: {M('lambda2'):.3f} | " if len(lists['Hrate_model']) else "") + \
            (f"MI@2: {M('MI_lag2'):.2f}b | AvgRun: {M('AvgRun'):.2f} | effK%: {100*M('EffKFrac'):.1f}% | " if len(lists['AvgRun']) else "") + \
            (f"AR1@1: {100*M('ar_top1_k1'):.1f}% | AR2@1: {100*M('ar_top1_k2'):.1f}% | AR3@1: {100*M('ar_top1_k3'):.1f}% | " if len(lists['ar_top1_k1']) or len(lists['ar_top1_k2']) or len(lists['ar_top1_k3']) else "") + \
            f"Corr: {M('corr'):.4f}"
        print(msg)
        print('Current lr: {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
        if not if_training:
            self.history['epoch'].append(epoch + 1)
            self.history['PPL1'].append(M('PPL1', default=float('nan')))
            self.history['PPL2'].append(M('PPL2', default=float('nan')))
            self.history['PPL3'].append(M('PPL3', default=float('nan')))
            self.history['AR1@1'].append(100 * M('ar_top1_k1', default=float('nan')))
            self.history['AR2@1'].append(100 * M('ar_top1_k2', default=float('nan')))
            self.history['AR3@1'].append(100 * M('ar_top1_k3', default=float('nan')))
            if self.plot_dir and ((epoch + 1) % self.plot_every == 0): self._plot_curves()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return (
            torch.tensor(lists['recon']).mean(),
            torch.tensor(lists['embedding']).mean(),
            torch.tensor(lists['commitment']).mean(),
            torch.tensor(lists['total']).mean(),
            torch.tensor(lists['corr']).mean(),
        )

    def _plot_curves(self):
        epochs = self.history['epoch']
        if len(epochs) == 0: return
        plt.figure()
        if any(not math.isnan(x) for x in self.history['PPL1']): plt.plot(epochs, self.history['PPL1'], label='PPL1')
        if any(not math.isnan(x) for x in self.history['PPL2']): plt.plot(epochs, self.history['PPL2'], label='PPL2')
        if any(not math.isnan(x) for x in self.history['PPL3']): plt.plot(epochs, self.history['PPL3'], label='PPL3')
        plt.xlabel('Epoch'); plt.ylabel('Perplexity'); plt.title('PPL curves'); plt.legend()
        out = os.path.join(self.plot_dir, 'ppl_curves.png')
        plt.savefig(out, bbox_inches='tight'); plt.close()
        plt.figure()
        if any(not math.isnan(x) for x in self.history['AR1@1']): plt.plot(epochs, self.history['AR1@1'], label='AR1@1')
        if any(not math.isnan(x) for x in self.history['AR2@1']): plt.plot(epochs, self.history['AR2@1'], label='AR2@1')
        if any(not math.isnan(x) for x in self.history['AR3@1']): plt.plot(epochs, self.history['AR3@1'], label='AR3@1')
        plt.xlabel('Epoch'); plt.ylabel('Top-1 accuracy (%)'); plt.title('AR@k top-1'); plt.legend()
        out = os.path.join(self.plot_dir, 'ar_top1_curves.png')
        plt.savefig(out, bbox_inches='tight'); plt.close()

    def train(self):
        if self.epoch_warmup > 0: 
            for epoch in range(self.epoch_warmup): self.process_warmup(epoch)
        self.model.initialize_codebook_with_kmeans(self.train_loader, device=self.device)
        for epoch in range(self.epoch_max):
            self.process(epoch, if_training=True)
            if self.recluster_interval_epoch > 0 and (epoch + 1) % self.recluster_interval_epoch == 0 and (epoch+1) <= self.recluster_end_epoch:
                recluster_success = self.model.perform_kmeans_recluster_and_reset_ema(initial_ema_count=self.initial_ema_count_after_recluster)
                if recluster_success: self.model.clear_buffer()
            if (epoch + 1) % self.evaluation_interval == 0: self.process(epoch, if_training=False)
