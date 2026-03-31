import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from util.distributed import _dist_avail_and_initialized, get_local_rank, get_rank, is_main_process, log_print
from .dt_layers import *
from .dt_utility import *
from .dt_utility import _infer_emb_sizes_from_state_dict, _maybe_strip_module_prefix, _as_trial_list, _pearson_corr_torch, _align_gt_to_stride_mod4_like_training, _pearson_corr_1d

from model.neural_quantizer.VQ_quantizer import VQForCalcium


def masked_ce(logits_BNTV, targets_BNT, mask_BNT, weight=None):
    B, N, T, V = logits_BNTV.shape
    logits = logits_BNTV.reshape(-1, V)
    tars = targets_BNT.reshape(-1)
    m = mask_BNT.reshape(-1).to(logits.dtype)
    ce = F.cross_entropy(logits, tars, reduction="none", weight=weight)
    num = m.sum().clamp_min(1.0)
    return (ce * m).sum() / num


def _autocast_ctx(device: str, use_bf16: bool = True):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if use_bf16 else torch.float16))
    return nullcontext()

class AxialAR(nn.Module):
    def __init__(self, cfg: AxialARCFG):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.neu_emb = nn.Embedding(cfg.n_neurons, cfg.d_model)
        self.ses_emb = nn.Embedding(cfg.n_sessions, cfg.d_model)   # NEW
        self.time_emb = nn.Embedding(8192, cfg.d_model) if cfg.use_abs_time_emb else None
        self.drop = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList([AxialARBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNormFP32(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        for m in self.modules(): 
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.head.weight = self.tok_emb.weight
        self.mtp_heads = None

    def encode(self, tokens_inBNT, neuron_idsBNT, time_posBT, time_kpmaskBT, session_idB=None):
        x = self.tok_emb(tokens_inBNT) + self.neu_emb(neuron_idsBNT)
        if session_idB is not None:
            ses = self.ses_emb(session_idB)
            x = x + ses[:, None, None, :]
        if self.time_emb is not None: x = x + self.time_emb(time_posBT).unsqueeze(1)
        x = self.drop(x)
        for blk in self.blocks: x = blk(x, time_posBT, time_kpmaskBT)
        x = self.ln_f(x)
        return x

    def forward(self, tokens_inBNT, neuron_idsBNT, time_posBT, time_kpmaskBT, session_idB=None, return_h: bool=False):
        h = self.encode(tokens_inBNT, neuron_idsBNT, time_posBT, time_kpmaskBT, session_idB=session_idB)
        logits = self.head(h)
        return (logits, h) if return_h else logits

    def build_mtp_heads(self, H: int, d_hidden: int = None):
        D = self.cfg.d_model; V = self.cfg.vocab
        Dh = D if d_hidden is None else int(d_hidden)
        heads = nn.ModuleList()
        for _ in range(H-1):
            heads.append(nn.Sequential(
                nn.Linear(D, Dh, bias=False),
                nn.GELU(),
                nn.Linear(Dh, V, bias=False)
            ))
        self.mtp_heads = heads
        self.mtp_heads.to(self.tok_emb.weight.device)

def _expand_embedding_(emb: torch.nn.Embedding, new_rows: int, init="normal"):
    if new_rows <= 0: return emb
    old_w = emb.weight.data
    old_n, d = old_w.shape
    new_w = old_w.new_empty((old_n + new_rows, d))
    new_w[:old_n] = old_w
    if init == "zeros": new_w[old_n:] = 0.0
    elif init == "mean":
        mu = old_w.mean(dim=0, keepdim=True)
        new_w[old_n:] = mu + 0.01 * torch.randn_like(new_w[old_n:])
    else: new_w[old_n:] = 0.02 * torch.randn_like(new_w[old_n:])
    emb.weight = torch.nn.Parameter(new_w, requires_grad=emb.weight.requires_grad)
    emb.num_embeddings = old_n + new_rows
    return emb

def expand_embeddings_for_new_sessions(
    model: AxialAR, 
    new_n_sessions: int, 
    new_n_neurons: int,
    init_session="mean", init_neuron="normal",
    freeze_old: bool = False
):
    device = model.neu_emb.weight.device
    old_s = model.ses_emb.num_embeddings
    old_n = model.neu_emb.num_embeddings
    add_s = max(0, new_n_sessions - old_s)
    add_n = max(0, new_n_neurons  - old_n)
    if add_s > 0: _expand_embedding_(model.ses_emb, add_s, init=init_session)
    if add_n > 0: _expand_embedding_(model.neu_emb, add_n, init=init_neuron)
    model.to(device)
    if freeze_old:
        if add_s > 0:
            def hook_s(grad):
                grad[:old_s].zero_()
                return grad
            model.ses_emb.weight.register_hook(hook_s)
        if add_n > 0:
            def hook_n(grad):
                grad[:old_n].zero_()
                return grad
            model.neu_emb.weight.register_hook(hook_n)
    model.cfg.n_sessions = model.ses_emb.num_embeddings
    model.cfg.n_neurons  = model.neu_emb.num_embeddings
    return model

@torch.no_grad()
def init_mtp_heads_from_base_head(model: "AxialAR"):
    assert model.mtp_heads is not None
    W = model.head.weight.detach()  # [V,D]
    V, D = W.shape
    for h in model.mtp_heads:
        lin1: nn.Linear = h[0]
        lin2: nn.Linear = h[2]
        Dh = lin1.out_features
        lin1.weight.zero_()
        m = min(Dh, D)
        lin1.weight[:m, :m].copy_(torch.eye(m, device=lin1.weight.device, dtype=lin1.weight.dtype))
        lin2.weight.zero_()
        if lin2.in_features == D: lin2.weight.copy_(W.to(lin2.weight.dtype))
        elif lin2.in_features > D: lin2.weight[:, :D].copy_(W.to(lin2.weight.dtype))
        else: lin2.weight.copy_(W[:, :lin2.in_features].to(lin2.weight.dtype))
        
def build_ar_batch(batch, cfg: AxialARCFG, training: bool):
    xBNT = batch["xBNT"]
    lengths = batch["lengths"]
    session_idB = batch["session_idB"]
    neuron_idsBN = batch["neuron_idsBN"]
    B, N, Tm = xBNT.shape
    device = xBNT.device
    if (not training and cfg.eval_use_full_trial) or cfg.train_window_T == 0:
        Tw = Tm
        time_posBT = torch.arange(Tw, device=device).unsqueeze(0).expand(B, Tw)
        time_kpmaskBT = (torch.arange(Tw, device=device).unsqueeze(0) < lengths.reshape(B, 1))
    else:
        Tw = int(min(cfg.train_window_T, int(lengths.min().item())))
        if Tw < 2:
            Tw = 2
            x2 = torch.zeros((B, N, Tw), dtype=xBNT.dtype, device=device)
            t_fill = min(Tm, Tw)
            if t_fill > 0: x2[:, :, :t_fill] = xBNT[:, :, :t_fill]
            xBNT = x2
            time_posBT = torch.arange(Tw, device=device).unsqueeze(0).expand(B, Tw)
            time_kpmaskBT = torch.zeros((B, Tw), dtype=torch.bool, device=device)
            time_kpmaskBT[:, 0] = True
        else:
            starts = []
            for b in range(B):
                Tb = int(lengths[b].item())
                hi = Tb - Tw + 1
                s = 0 if hi <= 1 else int(torch.randint(0, hi, (1,), device=device))
                starts.append(s)
            starts = torch.tensor(starts, dtype=torch.long, device=device)
            idx_t = torch.arange(Tw, device=device).unsqueeze(0) + starts.unsqueeze(1)
            xBNT = xBNT.gather(2, idx_t.unsqueeze(1).expand(B, N, Tw))
            time_posBT = torch.arange(Tw, device=device).unsqueeze(0).expand(B, Tw)
            time_kpmaskBT = torch.ones((B, Tw), dtype=torch.bool, device=device)
    tokens_inBNT = xBNT[:, :, :-1]
    targetsBNT   = xBNT[:, :,  1:]
    time_pos_inBT = time_posBT[:, :-1]
    time_kpmask_inBT = time_kpmaskBT[:, :-1]
    target_validBT   = time_kpmaskBT[:, 1:]                    # [B, T_in]
    T_in = tokens_inBNT.shape[-1]
    neuron_idsBNT = neuron_idsBN[:, :, None].expand(B, N, T_in)
    target_validBNT = target_validBT[:, None, :].expand(B, N, T_in)
    return tokens_inBNT, targetsBNT, neuron_idsBNT, time_pos_inBT, time_kpmask_inBT, target_validBNT, session_idB

def build_scheduler(
    optimizer, 
    total_steps: int,
    warmup_ratio: float = 0.06,
    base_lr: float = 2e-4,
    min_lr: float = 1e-6,
    kind: str = "cosine"
):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr_ratio = max(0.0, min(1.0, min_lr / max(1e-12, base_lr)))
    def lr_lambda(step: int):
        if step < warmup_steps: return (step + 1) / warmup_steps
        if total_steps <= warmup_steps: progress = 1.0
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
        if kind == "linear": return 1.0 - (1.0 - min_lr_ratio) * progress
        elif kind == "cosine": return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else: return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def ar_teacher_forced_generate_trials(
    model: nn.Module,                            
    trials: List[np.ndarray],
    device: str,
    prefix_len: int = 10,
    out_npz_path: str = "ar_tf.npz",
    use_ema: bool = False,
    ema: Optional["EMAHelper"] = None,
    temp: float = 1.0,
    topk: int = 0
):
    if use_ema and (ema is not None): ema.apply_shadow(model)
    model.eval()
    saved = {}
    for i, arr in enumerate(trials):
        arr = np.asarray(arr, dtype=np.int64)
        N, T = arr.shape
        if T <= 1:
            saved[f"trial_{i:05d}"] = arr.copy()
            continue
        P = min(int(prefix_len), T)
        if P < 0: raise ValueError(f"prefix_len must be >=0, got {prefix_len}")
        tokens_in = torch.from_numpy(arr[:, :-1]).long().to(device)   # [N, T-1]
        cur_len = tokens_in.size(1)                                   # T-1
        time_pos   = torch.arange(cur_len, device=device).unsqueeze(0)               # [1, T-1]
        kpmask     = torch.ones((1, cur_len), dtype=torch.bool, device=device)       # [1, T-1]
        neuron_ids = torch.arange(N, device=device).reshape(1, N, 1).expand(1, N, cur_len)  # [1, N, T-1]
        logits = model(tokens_in.unsqueeze(0), neuron_ids, time_pos, kpmask).squeeze(0)  # [N, T-1, V]
        logits = logits / max(1e-6, float(temp))
        V = int(logits.size(-1))
        if topk > 0:
            kk = int(min(topk, V))
            topv, topi = logits.topk(k=kk, dim=-1)        # [N, T-1, kk]
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, topi, topv)
        pred_next = logits.argmax(dim=-1).detach().cpu().numpy()  # [N, T-1]
        out = arr.copy()
        if P < T:
            if P == 0: out[:, 1:T] = pred_next[:, 0:(T-1)]
            else: out[:, P:T] = pred_next[:, (P-1):(T-1)]
        saved[f"trial_{i:05d}"] = out
    if use_ema and (ema is not None): ema.restore(model)
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez(out_npz_path, **saved)
    return out_npz_path

@torch.no_grad()
def ar_greedy_generate_trials(
    model: nn.Module,
    trials: List[np.ndarray],
    vocab: int,
    device: str,
    prefix_len: int = 15,
    out_npz_path: str = "ar_gen.npz",
    use_ema=False, 
    ema: Optional["EMAHelper"]=None,
    temp: float = 1.0, 
):
    print('Here call ar_greedy_generate_trials')
    if use_ema and ema is not None: ema.apply_shadow(model)
    model.eval()
    saved = {}
    for i, arr in enumerate(trials):
        N, T = arr.shape
        P = min(prefix_len, T)
        gen = torch.from_numpy(arr[:, :P].copy()).long().to(device)
        while gen.size(1) < T:
            cur_len = gen.size(1)
            tokens_in = gen.unsqueeze(0)
            time_pos  = torch.arange(cur_len, device=device).unsqueeze(0)
            kpmask    = torch.ones((1, cur_len), dtype=torch.bool, device=device)
            neuron_ids= torch.arange(N, device=device).reshape(1,N,1).expand(1,N,cur_len)
            logits = model(tokens_in, neuron_ids, time_pos, kpmask)[:, :, -1, :]
            logits = logits / max(1e-6, float(temp))
            next_tok = logits.argmax(dim=-1).squeeze(0)
            gen = torch.cat([gen, next_tok.unsqueeze(1)], dim=1)
        saved[f"trial_{i:05d}"] = gen[:, :T].detach().cpu().numpy()
    if use_ema and ema is not None: ema.restore(model)
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez(out_npz_path, **saved)
    return out_npz_path

@torch.no_grad()
def ar_beam_generate_trials(
    model,
    trials,
    device,
    prefix_len=15,
    out_npz_path="ar_beam.npz",
    beam_size=2,
    lambda_emb=0.0,
    bigram_logP=None,
    alpha_bigram=0.0,
    temp=1.0,
    post_smooth_win: int = 0,
    use_ema=False,
    ema=None,
    codebook_E: torch.Tensor = None,
    beam_expand_neurons: int = 64,
    beam_per_neuron_topk: int = 32,
    debug_csv: int = 0,
    debug_dir: Optional[str] = None,
    debug_topk: int = 10,          # dump topK per neuron as list columns
    debug_max_trials: int = 16,    # only dump first K trials
    debug_dump_hypotheses: int = 0 # dump beam_hypotheses.csv (can be big)
):
    print(f"[BeamGen] temp={temp} beam={beam_size} k={beam_per_neuron_topk} expandN={beam_expand_neurons}")
    model.eval()
    beam_size = int(beam_size)
    if beam_size <= 0: raise ValueError(f"beam_size must be >=1, got {beam_size}")
    if use_ema and ema is not None: ema.apply_shadow(model)
    if bigram_logP is not None:
        if not torch.is_tensor(bigram_logP): bigram_logP = torch.tensor(bigram_logP)
        bigram_logP = bigram_logP.to(device=device, dtype=torch.float32)
    E = None
    need_E = (lambda_emb and lambda_emb > 0.0) or (post_smooth_win and post_smooth_win > 1)
    if need_E and (codebook_E is not None): E = F.normalize(codebook_E.to(device=device), dim=-1)  # [V,D]
    f_step = f_neu = f_hyp = None
    w_step = w_neu = w_hyp = None
    def _close_writers():
        for f in (f_step, f_neu, f_hyp):
            try:
                if f is not None: f.close()
            except Exception: pass
    def _logp_for_seq(seq_NL: torch.Tensor) -> torch.Tensor:
        N = int(seq_NL.size(0))
        L = int(seq_NL.size(1))
        tokens_in = seq_NL.unsqueeze(0)  # [1,N,L]
        time_pos  = torch.arange(L, device=device).unsqueeze(0)  # [1,L]
        kpmask    = torch.ones((1, L), dtype=torch.bool, device=device)
        neuron_ids= torch.arange(N, device=device).reshape(1, N, 1).expand(1, N, L)
        logits = model(tokens_in, neuron_ids, time_pos, kpmask)[:, :, -1, :]  # [1,N,V]
        logits = logits / max(1e-6, float(temp))
        return F.log_softmax(logits, dim=-1).squeeze(0)  # [N,V]
    saved: Dict[str, np.ndarray] = {}
    try:
        for i, arr in enumerate(trials):
            arr = np.asarray(arr, dtype=np.int64)
            if arr.ndim != 2: raise ValueError(f"Each trial must be [N,T], got shape {arr.shape}")
            N, T = arr.shape
            if T <= 1:
                saved[f"trial_{i:05d}"] = arr.copy()
                continue
            P = min(int(prefix_len), T)
            if P < 0: raise ValueError(f"prefix_len must be >=0, got {prefix_len}")
            if P == 0: P = 1
            prefix = torch.from_numpy(arr[:, :P].copy()).long().to(device)  # [N,P]
            beams = [{'seq': prefix.clone(), 'score': 0.0}]  # each seq: [N,cur_len]
            for t in range(P, T):
                new_beams = []
                for parent_id, b in enumerate(beams):
                    seq = b["seq"]  # [N,cur_len]
                    cur_len = int(seq.size(1))
                    logp = _logp_for_seq(seq)  # [N,V]
                    V = int(logp.size(-1))
                    k_cand = int(beam_per_neuron_topk or 16)
                    k_cand = int(min(max(2, k_cand), V))
                    topv, topi = torch.topk(logp, k=k_cand, dim=-1)  # [N,k]
                    cands = topi
                    scores = topv.clone()
                    if (lambda_emb and lambda_emb > 0.0) and (E is not None) and (cur_len > 0):
                        prev = seq[:, -1]      # [N]
                        prev_e = E[prev]       # [N,D]
                        cand_e = E[cands]      # [N,k,D]
                        dist2 = ((cand_e - prev_e.unsqueeze(1)) ** 2).sum(-1)  # [N,k]
                        scores = scores - float(lambda_emb) * dist2
                    if (bigram_logP is not None) and (alpha_bigram and alpha_bigram > 0.0) and (cur_len > 0):
                        prev = seq[:, -1]  # [N]
                        bg = bigram_logP[prev.unsqueeze(1), cands]  # [N,k]
                        scores = scores + float(alpha_bigram) * bg
                    if beam_expand_neurons and 0 < int(beam_expand_neurons) < N:
                        probs = logp.exp()
                        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)  # [N]
                        sel = torch.topk(ent, k=int(beam_expand_neurons)).indices
                    else: sel = torch.arange(N, device=device)
                    joint = topk_joint_from_scores(scores.index_select(0, sel), beam_size)
                    if not joint: continue
                    best_idx_all = scores.argmax(-1)  # [N] (0..k-1)
                    for child_rank, (cols_sel, _sumv) in enumerate(joint):
                        idx_choose = best_idx_all.clone()
                        for j, n_id in enumerate(sel): idx_choose[n_id] = int(cols_sel[j])
                        next_tokens = cands[torch.arange(N, device=device), idx_choose]  # [N]
                        joint_score = scores[torch.arange(N, device=device), idx_choose].sum().item()
                        new_seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)
                        new_beams.append({
                            "seq": new_seq,
                            "score": b["score"] + joint_score,
                            "parent_id": parent_id,
                            "child_rank": child_rank
                        })
                if not new_beams: new_beams = beams
                new_beams.sort(key=lambda x: x["score"], reverse=True)
                beams = new_beams[:beam_size]
            gen = max(beams, key=lambda d: d["score"])["seq"]  # [N,T]
            if post_smooth_win and post_smooth_win > 1 and E is not None:
                w = int(post_smooth_win)
                if w % 2 == 0: w += 1
                pad = w // 2
                tok = gen.detach()     # [N,T]
                emb = E[tok]           # [N,T,D]
                x = emb.transpose(1, 2).contiguous()  # [N,D,T]
                kernel = torch.ones((emb.size(-1), 1, w), device=emb.device, dtype=emb.dtype) / float(w)
                sm = torch.empty_like(emb)
                for n in range(N):
                    sm_n = F.conv1d(x[n].unsqueeze(0), kernel, padding=pad, groups=emb.size(-1))
                    sm[n] = sm_n.squeeze(0).transpose(0, 1)
                sm = F.normalize(sm, dim=-1)        # [N,T,D]
                En = F.normalize(E, dim=-1)         # [V,D]
                sim = torch.einsum('ntd,vd->ntv', sm, En)  # [N,T,V]
                gen2 = sim.argmax(dim=-1)           # [N,T]
                gen2[:, :P] = prefix[:, :P]
                gen = gen2
            saved[f"trial_{i:05d}"] = gen[:, :T].detach().cpu().numpy()
    finally: _close_writers()
    if use_ema and ema is not None: ema.restore(model)
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez(out_npz_path, **saved)
    return out_npz_path

@torch.no_grad()
def topk_joint_from_scores(scores_mk: torch.Tensor, beam_size: int):
    beam_size = int(beam_size)
    if beam_size <= 0: return []
    if scores_mk.ndim != 2: raise ValueError(f"scores_mk must be 2D [m,k], got shape {tuple(scores_mk.shape)}")
    m, k = scores_mk.shape
    if m == 0 or k == 0: return []
    scores_cpu = scores_mk.detach().to("cpu")
    vals, idxs = torch.sort(scores_cpu, descending=True, dim=1)   # [m,k]
    vals_np = vals.numpy()
    idxs_np = idxs.numpy()
    if beam_size == 1:
        cols = [int(idxs_np[r, 0]) for r in range(m)]
        sumv = float(vals_np[:, 0].sum())
        return [(cols, sumv)]
    start_choice = tuple([0] * m)  # choose best (rank-0) in every row
    start_sum = float(vals_np[:, 0].sum())
    heap = [(-start_sum, start_choice)]
    visited = {start_choice}
    results = []
    while heap and len(results) < beam_size:
        neg_sum, choice = heapq.heappop(heap)
        cur_sum = -neg_sum
        results.append((choice, cur_sum))
        for r in range(m):
            cr = choice[r]
            nr = cr + 1
            if nr >= k: continue
            nxt = list(choice)
            nxt[r] = nr
            nxt = tuple(nxt)
            if nxt in visited: continue
            new_sum = cur_sum - float(vals_np[r, cr]) + float(vals_np[r, nr])
            heapq.heappush(heap, (-new_sum, nxt))
            visited.add(nxt)
    out = []
    for choice, sumv in results:
        cols = [int(idxs_np[r, choice[r]]) for r in range(m)]
        out.append((cols, float(sumv)))
    return out

def build_vq_decoder(
    vq_state_path: str,
    discretization_window:int, 
    overlap_window:int,
    n_emb:int, 
    dim_emb:int, 
    heads:int,
    enc_layers:int, 
    dec_layers:int,
    encoder_causal: bool, 
    decoder_causal: bool, 
    device: str, 
    lookahead_tokens: int
):
    m = VQForCalcium(discretization_window, overlap_window, n_emb, dim_emb, heads,                  
        enc_layers, dec_layers, dropout_ratio=.2, decay=0.99, epsilon=1e-5, use_gumbel=False,
        use_gumbel_hard=False, temperature=1.0, reset_threshold_ratio=0.0, dead_code_ema_reset_val =1.,
        use_periodic_kmeans_recluster=True, z_e_buffer_capacity=10000, min_z_e_for_recluster=1000, temperature_entropy=1.0,
        encoder_causal=encoder_causal, decoder_causal=decoder_causal, lookahead_tokens=lookahead_tokens, max_ar_k=4)
    m.to(device).eval()
    m = torch.compile(m)
    sd = torch.load(vq_state_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict): sd = sd['state_dict']
    sd = _maybe_strip_module_prefix(sd)
    m.load_state_dict(sd, strict=True)
    return m, overlap_window

@torch.no_grad()
def vq_decode_tokens_to_frames(model_vq, tokens_nt: np.ndarray, device: str="cuda") -> np.ndarray:
    tok = torch.as_tensor(tokens_nt, dtype=torch.long, device=device)
    z_q = model_vq.codebook(tok)  # usually [N,Ttok,D]
    try: dec = model_vq.decoding(z_q)             
    except Exception: dec = model_vq.decoding(z_q.unsqueeze(1)) 
    if dec.dim() == 3: dec = dec.squeeze(1)
    return dec.detach().cpu().numpy()

@torch.no_grad()
def greedy_rollout_tokens_one(
    model,
    tok_gt_NT: torch.Tensor,          # [N,T] long
    neuron_ids_N: torch.Tensor,       # [N] long (global neuron ids)
    session_id: torch.Tensor,         # scalar tensor
    prefix_tokens: int,
    device: str = "cuda",
):
    N, T = tok_gt_NT.shape
    if T <= 1: return tok_gt_NT.clone()
    P = int(max(1, min(prefix_tokens, T - 1)))   # keep at least 1 token
    gen = tok_gt_NT[:, :P].clone()               # [N,P]
    neuron_ids_N = neuron_ids_N.to(device)
    sid = session_id.reshape(1).to(device)
    for _ in range(P, T):
        curT = gen.shape[1]                      # context length
        x_in = gen.unsqueeze(0)                  # [1,N,curT]
        time_pos = torch.arange(curT, device=device).unsqueeze(0)  # [1,curT]
        kpmask = torch.ones((1, curT), dtype=torch.bool, device=device)
        neuron_idsBNT = neuron_ids_N.view(1, N, 1).expand(1, N, curT)  # [1,N,curT]
        logits = model(x_in, neuron_idsBNT, time_pos, kpmask, session_idB=sid)  # [1,N,curT,V]
        nxt = logits[0, :, -1].argmax(dim=-1)     # [N]
        gen = torch.cat([gen, nxt.unsqueeze(1)], dim=1)
    return gen

@torch.no_grad()
def eval_multisession_val_vq_recon_rollout(
    model,
    dl_va,
    meta: dict,
    vq_model=None,
    vq_stride: int = 4,
    prefix_tokens: int = 0,
    device: str = "cuda",
    use_ema: bool = False,
    ema: Optional["EMAHelper"] = None,
    print_per_trial: bool = False,
    tag: str = "VAL",
    trace_key: str = "trace",       
    split: Optional[str] = None,      
    id_offset_neuron: int = 0,
    id_offset_session: int = 0,
    max_batches: Optional[int] = None,
    log_txt_path: Optional[str] = None,
):
    if _dist_avail_and_initialized(): dist.barrier()
    log_print(f"[RANK {get_rank()}] start evaluation.", log_txt_path=log_txt_path, force=True)
    if vq_model is None: raise ValueError("eval_multisession_val_vq_recon_rollout: vq_model is required.")
    model.eval()
    if use_ema and (ema is not None): ema.apply_shadow(model)
    if split is None:
        ds = getattr(dl_va, "dataset", None)
        split = getattr(ds, "split", None)
        if split is None:
            u = str(tag).upper()
            split = "val" if u.startswith("VAL") else ("test" if u.startswith("TEST") else "val")
    sid2name: Dict[int, str] = {}
    sid2path: Dict[int, str] = {}
    sid2offset: Dict[int, int] = {}
    if meta is not None and "sessions" in meta:
        for s in meta["sessions"]:
            sid = int(s["session_id"])
            sid2path[sid] = s["path"]
            sid2name[sid] = os.path.basename(s["path"])
            sid2offset[sid] = int(s.get("neuron_offset", 0))
    ds_obj = getattr(dl_va, "dataset", None)
    if (ds_obj is not None) and hasattr(ds_obj, "sessions"):
        for s in getattr(ds_obj, "sessions"):
            sid = int(s.get("session_id", -1))
            if sid < 0: continue
            sid2path.setdefault(sid, s["path"])
            sid2name.setdefault(sid, os.path.basename(s["path"]))
            sid2offset.setdefault(sid, int(s.get("neuron_offset", 0)))
    trace_cache: Dict[int, List[np.ndarray]] = {}
    def _load_trace_trials_for_sid(sid: int) -> List[np.ndarray]:
        if sid in trace_cache:
            return trace_cache[sid]
        if sid not in sid2path:
            raise KeyError(f"[Eval] session_id={sid} not found in meta['sessions']; cannot locate npz path.")
        p = sid2path[sid]
        def _is_index_dict(d):
            if not isinstance(d, dict) or len(d) == 0:
                return False
            ks = list(d.keys())
            return all(str(k).isdigit() for k in ks)
        def _sort_keys(ks):
            return sorted(ks, key=lambda x: int(str(x)) if str(x).isdigit() else str(x))
        def _append_trace_from_trial_obj(obj, out_list):
            obj = _unwrap_object(obj)
            if isinstance(obj, dict) and (trace_key not in obj) and _is_index_dict(obj):
                for k in _sort_keys(obj.keys()):
                    _append_trace_from_trial_obj(obj[k], out_list)
                return
            if isinstance(obj, dict):
                if trace_key not in obj:
                    raise KeyError(
                        f"[Eval] {p} split='{split}' trial dict missing key='{trace_key}'. "
                        f"trial keys={list(obj.keys())[:20]}"
                    )
                arr = obj[trace_key]
            else:
                arr = obj
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"[Eval] trace must be 2D [N,T], got shape={arr.shape} in {p} split={split}"
                )
            out_list.append(arr)
        with np.load(p, allow_pickle=True) as z:
            if split not in z.files:
                raise KeyError(f"[Eval] {p} missing split='{split}'. keys={list(z.files)[:20]}")
            obj = z[split]
            out: List[np.ndarray] = []
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                if obj.shape == () and isinstance(obj.item(), dict):
                    _append_trace_from_trial_obj(obj.item(), out)
                else:
                    for item in obj:
                        _append_trace_from_trial_obj(item, out)
            else:
                _append_trace_from_trial_obj(obj, out)
        trace_cache[sid] = out
        return out
    from collections import defaultdict
    sum_corr_by_sid: Dict[int, float] = defaultdict(float)
    sum_mse_by_sid:  Dict[int, float] = defaultdict(float)
    cnt_by_sid:      Dict[int, int]   = defaultdict(int)
    agg_sids: List[int] = []
    if meta is not None and isinstance(meta, dict) and ("sessions" in meta) and isinstance(meta["sessions"], (list, tuple)):
        try: agg_sids = sorted({int(s["session_id"]) for s in meta["sessions"] if isinstance(s, dict) and ("session_id" in s)})
        except Exception: agg_sids = []
    if not agg_sids: agg_sids = sorted({int(k) for k in sid2name.keys()})
    sid2idx = {sid: i for i, sid in enumerate(agg_sids)}
    for bi, batch in enumerate(dl_va):
        if (max_batches is not None) and (bi >= int(max_batches)): break
        xBNT        = batch["xBNT"].to(device, non_blocking=True)               # [B,Nsub,Tmax]
        lengths     = batch["lengths"].to(device, non_blocking=True)            # [B]
        session_idB_raw = batch["session_idB"].to(device, non_blocking=True)        # [B]
        neuron_idsBN_raw = batch["neuron_idsBN"].to(device, non_blocking=True)       # [B,Nsub]
        session_idB = session_idB_raw + int(id_offset_session) if id_offset_session else session_idB_raw
        neuron_idsBN = neuron_idsBN_raw + int(id_offset_neuron) if id_offset_neuron else neuron_idsBN_raw
        trial_idxB  = batch.get("trial_idxB", None)
        if trial_idxB is not None: trial_idxB = trial_idxB.to(device, non_blocking=True)
        else: raise KeyError("[Eval] trial_idxB missing in batch; cannot index raw trace from npz.")
        B, Nsub, Tmax = xBNT.shape
        for b in range(B):
            L = int(lengths[b].item())
            if L <= 1: continue
            tok_gt = xBNT[b, :, :L]  # [Nsub,L]
            sid = int(session_idB[b].item())
            sid_trace = int(session_idB_raw[b].item())
            tid = int(trial_idxB[b].item())
            tok_pred = greedy_rollout_tokens_one(
                model=model,
                tok_gt_NT=tok_gt,
                neuron_ids_N=neuron_idsBN[b],
                session_id=session_idB[b],
                prefix_tokens=prefix_tokens,
                device=device,
            )
            pr_frames = vq_decode_tokens_to_frames(vq_model, tok_pred.detach().cpu().numpy(), device=device)  # [Nsub, Tdec]
            trace_trials = _load_trace_trials_for_sid(sid_trace)
            if tid < 0 or tid >= len(trace_trials): raise IndexError(f"[Eval] trial_idx={tid} out of range for sid={sid} split={split} n={len(trace_trials)}")
            gt_full = trace_trials[tid]  # [N_session, Ttrue]
            off = int(sid2offset.get(sid_trace, 0))
            local_idx = (neuron_idsBN_raw[b].detach().cpu().numpy() - off).astype(np.int64)
            if (local_idx.min() < 0) or (local_idx.max() >= gt_full.shape[0]):
                raise IndexError(
                    f"[Eval] neuron index OOB after offset correction: sid={sid} off={off} "
                    f"local_min={local_idx.min()} local_max={local_idx.max()} gt_full_N={gt_full.shape[0]}"
                )
            gt_raw = gt_full[local_idx, :]  # [Nsub, Ttrue]
            if int(vq_stride) == 4: gt_frames = _align_gt_to_stride_mod4_like_training(gt_raw, stride=int(vq_stride))
            else: gt_frames = gt_raw
            ignore = int(prefix_tokens + 1) * int(vq_stride)
            Tcmp = int(min(gt_frames.shape[1], pr_frames.shape[1]))
            if Tcmp <= ignore + 2: continue
            gt = torch.from_numpy(gt_frames[:, ignore:Tcmp]).to(device)
            pr = torch.from_numpy(pr_frames[:, ignore:Tcmp]).to(device)
            mse  = torch.mean((gt - pr) ** 2).item()
            corr = _pearson_corr_torch(gt, pr).mean().item()
            sum_corr_by_sid[sid] += float(corr)
            sum_mse_by_sid[sid]  += float(mse)
            cnt_by_sid[sid]      += 1
            if print_per_trial and is_main_process():
                name = sid2name.get(sid, str(sid))
                log_print(f"[{tag}] session={sid:03d} ({name}) trial={tid:05d} corr={corr:.4f} mse={mse:.6g}", log_txt_path=log_txt_path, force=True)
    S = len(agg_sids)
    red_device = device if (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    vec = torch.zeros((3 * S,), device=red_device, dtype=torch.float64)
    for sid, idx in sid2idx.items():
        vec[idx]         = float(sum_corr_by_sid.get(sid, 0.0))
        vec[S + idx]     = float(sum_mse_by_sid.get(sid,  0.0))
        vec[2 * S + idx] = float(cnt_by_sid.get(sid,      0))
    if _dist_avail_and_initialized(): dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    sum_corr_vec = vec[0:S]
    sum_mse_vec  = vec[S:2*S]
    cnt_vec      = vec[2*S:3*S]
    if is_main_process():
        per_sess_corr = []
        per_sess_mse  = []
        total_cnt = float(cnt_vec.sum().item())
        if total_cnt <= 0: log_print(f"[{tag}] No valid trials were evaluated.\n", log_txt_path=log_txt_path, force=True)
        else:
            for i, sid in enumerate(agg_sids):
                n = int(cnt_vec[i].item())
                if n <= 0: continue
                name = sid2name.get(sid, str(sid))
                c = float((sum_corr_vec[i] / cnt_vec[i]).item())
                m = float((sum_mse_vec[i]  / cnt_vec[i]).item())
                log_print(f"  - session={sid:03d} ({name})  n_trials={n:4d}  corr={c:.4f}  mse={m:.6g}", log_txt_path=log_txt_path, force=True)
                per_sess_corr.append(c)
                per_sess_mse.append(m)
            micro_corr = float((sum_corr_vec.sum() / cnt_vec.sum().clamp_min(1.0)).item())
            micro_mse  = float((sum_mse_vec.sum()  / cnt_vec.sum().clamp_min(1.0)).item())
            macro_corr = float(np.mean(per_sess_corr)) if per_sess_corr else float("nan")
            macro_mse  = float(np.mean(per_sess_mse))  if per_sess_mse  else float("nan")
            log_print(f"[{tag}] Overall MICRO (avg over trials): corr={micro_corr:.4f} mse={micro_mse:.6g}", log_txt_path=log_txt_path, force=True)
            log_print(f"[{tag}] Overall MACRO (avg over sessions): corr={macro_corr:.4f} mse={macro_mse:.6g}\n", log_txt_path=log_txt_path, force=True)
    if _dist_avail_and_initialized(): dist.barrier()
    if use_ema and (ema is not None): ema.restore(model)


@torch.no_grad()
def eval_ar_npz_mean_corr(
    ar_npz_path: str,
    gt_X: np.ndarray,
    gt_L: np.ndarray,
    vq_model,
    stride: int,
    gen_prefix_tokens: int,
    device: str = "cuda"
) -> float:
    data = np.load(ar_npz_path, allow_pickle=False)
    keys = sorted([k for k in data.files if k.startswith("trial_")], key=lambda s: int(s.split("_")[-1]))
    B = min(len(keys), gt_X.shape[0])
    stride = int(stride)
    ignore_frames = int(gen_prefix_tokens) * stride
    per_trial = []
    for i in range(B):
        arr = data[keys[i]].astype(np.int64)  # [N, Ttok]
        N, _ = arr.shape
        if N != gt_X.shape[1]: raise ValueError(f"N mismatch at trial {i}: tokens.N={N} vs GT.N={gt_X.shape[1]}")
        pred_frames = vq_decode_tokens_to_frames(vq_model, arr, device=device)  # [N, Tdec]
        Tdec = int(pred_frames.shape[1])
        Ttrue = int(gt_L[i])
        gt_raw = gt_X[i, :, :Ttrue]  # [N, Ttrue]
        gt_frames = _align_gt_to_stride_mod4_like_training(gt_raw, stride=stride)  # [N, Talign]
        Talign = int(gt_frames.shape[1])
        Tcmp = min(Tdec, Talign)
        if Tcmp <= ignore_frames:
            per_trial.append(np.nan)
            continue
        s, e = ignore_frames, Tcmp
        gt_cmp = gt_frames[:, :Tcmp]
        pr_cmp = pred_frames[:, :Tcmp]
        corr_n = []
        for n in range(N): corr_n.append(_pearson_corr_1d(gt_cmp[n, s:e], pr_cmp[n, s:e]))
        corr_n = np.asarray(corr_n, dtype=np.float32)
        mean_corr = float(np.nanmean(corr_n)) if np.any(np.isfinite(corr_n)) else np.nan
        per_trial.append(mean_corr)
    per_trial = np.asarray(per_trial, dtype=np.float32)
    return float(np.nanmean(per_trial)) if per_trial.size else float('nan')

def run_axial_ar(
    train_trials: List[np.ndarray],
    val_trials: List[np.ndarray],
    vocab:int, 
    device:str="cuda",
    epochs:int=50, 
    lr:float=1e-4, 
    min_lr:float=1e-6,
    warmup_ratio:float=0.06, 
    lr_schedule:str="cosine",
    weight_decay:float=1e-3,
    gen_every:int=10, 
    gen_prefix:int=10, 
    gen_outdir:str="ar_gen",
    cfg_overrides:dict=None,
    vq_state: str = "",
    vq_disc_win: int = 4, 
    vq_overlap: int = 4,
    vq_n_emb: int = 128, 
    vq_dim_emb: int = 512,
    vq_heads: int = 4, 
    vq_enc_layers: int = 4, 
    vq_dec_layers: int = 4,
    vq_encoder_causal: bool = False, 
    vq_decoder_causal: bool = False,
    use_mtp: bool = False, 
    mtp_horizons: int = 3,
    mtp_hidden: int = 512, 
    mtp_weight: str = "geom", 
    mtp_geom_gamma: float = 0.7,
    mtp_warmup_epochs: int = 0, 
    mtp_lambda: float = 1.0,
    use_ema: bool = False, 
    ema_decay: float = 0.999, 
    ema_eval_only: bool = True,
    use_emb_consistency: bool = False, 
    emb_consis_lambda: float = 0.0, 
    emb_consis_apply_to_mtp: bool = False,
    use_emb_grad_ce: bool = False, 
    emb_grad_alpha: float = 0.7, 
    emb_grad_floor: float = 0.1,
    use_scheduled_sampling: bool = False, 
    ss_prob: float = 0.0, ss_block_len: int = 32,
    cf_counts: Optional[np.ndarray]=None, 
    cf_beta: float=0.5,
    neighbor_idx: Optional[object]=None, 
    tok_noise_p: float = 0.0, 
    neighbor_k:int=4,
    use_ngram_kd: bool = False, 
    ngram_order: int = 2, 
    ngram_alpha: float = 0.2, 
    ngram_laplace: float = 0.5,
    decoding_beam:int=1, 
    beam_lambda_emb_smooth:float=0.0, 
    use_bigram_rescore:bool=False, 
    bigram_logP:Optional[torch.Tensor]=None, 
    bigram_alpha:float=0.0,
    decode_temp:float=1.0, 
    debug_k:int=5, 
    debug_max_trials:int=16,
    ctx_analyze:int=0, 
    ctx_K:int=4,
    debug_detail: int=0, 
    debug_trial:int=0, 
    debug_topk:int=10,
    debug_token_dump:int=0, 
    post_smooth_win:int=0, 
    lookahead_tokens:int=1,
    beam_expand_neurons:int=1,
    beam_per_neuron_topk:int=1,
    ckpt_every: int = 10,
    ckpt_dir: Optional[str]=None,
    save_ema_weights: bool=True,
    dl_tr=None,
    dl_va=None,
    meta=None,
    compile_model: bool = False,
    compile_dynamic: bool = True,
    init_ckpt: Optional[str] = None,
    eval_trials_csv: Optional[str] = None,
    eval_detail_every: int = 5,
    id_offset_neuron: int = 0,
    id_offset_session: int = 0,
    init_use_ema_weights: bool = False,
    use_bf16: bool = True,
):
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    cfg_overrides = cfg_overrides or {}
    if dl_tr is not None:
        assert meta is not None, "When dl_tr is provided, meta must be provided (with n_neurons_total / n_sessions)."
        n_neurons_total = int(meta.get("n_neurons_total"))
        n_sessions = int(meta.get("n_sessions", 1))
    else:
        assert train_trials is not None and len(train_trials) > 0, "train_trials must be provided for single-session mode."
        n_neurons_total = int(train_trials[0].shape[0])
        n_sessions = 1
    if ckpt_dir is None: ckpt_dir = os.path.join(gen_outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_txt_path = os.path.join(ckpt_dir, "train_log.txt")
    if is_main_process():
        with open(log_txt_path, "w", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    target_n_neurons  = int(meta["n_neurons_total"]) if meta is not None else n_neurons_total
    target_n_sessions = int(meta["n_sessions"])      if meta is not None else n_sessions
    raw = None
    sd = None
    saved_cfg = None
    old_ckpt_n_neu, old_ckpt_n_ses = None, None
    if init_ckpt:
        raw, sd_use, sd_raw, sd_ema, saved_cfg = load_ckpt_state_dict(init_ckpt, prefer_ema=bool(init_use_ema_weights))
        sd = sd_use
        print(f"[Init] loaded ckpt={init_ckpt} prefer_ema={bool(init_use_ema_weights)} "
              f"keys(model)={len(sd)}")
        sd = normalize_state_dict_keys(sd)
        ckpt_cfg = saved_cfg if isinstance(saved_cfg, dict) else {}
        def _cfg_get(key, default): return ckpt_cfg.get(key, default)
        ckpt_n_neu, ckpt_n_ses = _infer_emb_sizes_from_state_dict(sd)
        ckpt_n_neu = int(ckpt_n_neu) if ckpt_n_neu is not None else int(target_n_neurons)
        ckpt_n_ses = int(ckpt_n_ses) if ckpt_n_ses is not None else int(target_n_sessions)
        n_layers_ckpt = int(_cfg_get("n_layers", cfg_overrides.get("n_layers", 6)))
        cfg = AxialARCFG(
            vocab=vocab,
            n_neurons=ckpt_n_neu,
            n_sessions=ckpt_n_ses,
            d_model=_cfg_get("d_model", cfg_overrides.get("d_model", 512)),
            n_layers=_cfg_get("n_layers", n_layers_ckpt),
            n_heads=_cfg_get("n_heads", cfg_overrides.get("n_heads", 8)),
            d_ff=_cfg_get("d_ff", cfg_overrides.get("d_ff", 2048)),
            emb_dropout=_cfg_get("emb_dropout", cfg_overrides.get("emb_dropout", 0.25)),
            attn_dropout=_cfg_get("attn_dropout", cfg_overrides.get("attn_dropout", 0.25)),
            dropout=_cfg_get("dropout", cfg_overrides.get("dropout", 0.25)),
            train_window_T=_cfg_get("train_window_T", cfg_overrides.get("train_window_T", 0)),
            eval_use_full_trial=_cfg_get("eval_use_full_trial", cfg_overrides.get("eval_use_full_trial", True)),
            use_abs_time_emb=_cfg_get("use_abs_time_emb", cfg_overrides.get("use_abs_time_emb", False)),
        )
        model_raw = AxialAR(cfg).to(device)
        if use_mtp and (getattr(model_raw, "mtp_heads", None) is None): model_raw.build_mtp_heads(mtp_horizons, d_hidden=mtp_hidden)
        missing, unexpected = model_raw.load_state_dict(sd, strict=False)
        if use_mtp and any(k.startswith("mtp_heads.") for k in missing): init_mtp_heads_from_base_head(model_raw)
        print(f"[InitCKPT] loaded {init_ckpt}; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        cfg = AxialARCFG(
            vocab=vocab,
            n_neurons=int(target_n_neurons),
            n_sessions=int(target_n_sessions),
            d_model=cfg_overrides.get("d_model", 512),
            n_layers=cfg_overrides.get("n_layers", 6),
            n_heads=cfg_overrides.get("n_heads", 8),
            d_ff=cfg_overrides.get("d_ff", 2048),
            emb_dropout=cfg_overrides.get("emb_dropout", 0.25),
            attn_dropout=cfg_overrides.get("attn_dropout", 0.25),
            dropout=cfg_overrides.get("dropout", 0.25),
            train_window_T=cfg_overrides.get("train_window_T", 0),
            eval_use_full_trial=cfg_overrides.get("eval_use_full_trial", True),
            use_abs_time_emb=cfg_overrides.get("use_abs_time_emb", False),
        )
        model_raw = AxialAR(cfg).to(device)
        if use_mtp and (getattr(model_raw, "mtp_heads", None) is None):
            model_raw.build_mtp_heads(mtp_horizons, d_hidden=mtp_hidden)
    if (model_raw.cfg.n_neurons < int(target_n_neurons)) or (model_raw.cfg.n_sessions < int(target_n_sessions)):
        print(f"[Expand] neu {model_raw.cfg.n_neurons}->{target_n_neurons} | ses {model_raw.cfg.n_sessions}->{target_n_sessions}")
        model_raw = expand_embeddings_for_new_sessions(
            model_raw,
            new_n_sessions=int(target_n_sessions),
            new_n_neurons=int(target_n_neurons),
            init_session="mean",
            init_neuron="normal",
            freeze_old=True,
        )
    train_mode = cfg_overrides.get("train_mode", "full")  # "full" or "embed_only"
    print('[train mode]: ', train_mode)
    for p in model_raw.parameters(): p.requires_grad = True
    if train_mode == "embed_only":
        for p in model_raw.parameters(): p.requires_grad = False
        model_raw.neu_emb.weight.requires_grad = True
        model_raw.ses_emb.weight.requires_grad = True
    trainable = [(n,p.numel()) for n,p in model_raw.named_parameters() if p.requires_grad]
    print("[Trainable]", len(trainable), "params, total =", sum(x[1] for x in trainable)/1e6, "M")
    print("Top10:", sorted(trainable, key=lambda x: -x[1])[:10])
    total_steps = epochs * max(1, len(dl_tr))
    model = model_raw
    if compile_model: model = torch.compile(model, dynamic=bool(compile_dynamic))
    if _dist_avail_and_initialized():
        lrk = get_local_rank()
        model = DDP(model, device_ids=[lrk], output_device=lrk, broadcast_buffers=False, find_unused_parameters=bool(use_mtp))
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        lname = name.lower()
        if name.endswith(".bias") or ("norm" in lname) or name.endswith("ses_emb.weight") or name.endswith("neu_emb.weight"): no_decay_params.append(p)
        else: decay_params.append(p)
    trainable_params = decay_params + no_decay_params
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )
    scheduler = build_scheduler(opt, total_steps, warmup_ratio, base_lr=lr, min_lr=min_lr, kind=lr_schedule)
    ema = EMAHelper(model, decay=ema_decay) if use_ema else None
    gt_X = gt_L = vq_model = vq_stride = None
    need_codebook = (
        use_emb_consistency or use_emb_grad_ce or (neighbor_idx is not None) or
        decoding_beam > 1 or beam_lambda_emb_smooth > 0.0 or (post_smooth_win and post_smooth_win > 1)
    )
    if vq_state:
        try:
            vq_model, vq_stride = build_vq_decoder(
                vq_state_path=vq_state,
                discretization_window=vq_disc_win, overlap_window=vq_overlap,
                n_emb=vq_n_emb, dim_emb=vq_dim_emb, heads=vq_heads,
                enc_layers=vq_enc_layers, dec_layers=vq_dec_layers,
                encoder_causal=vq_encoder_causal, decoder_causal=vq_decoder_causal, device=device, lookahead_tokens=lookahead_tokens,
            )
            print(f"[VQ] loaded; stride={vq_stride}")
        except Exception as e:
            print(f"[Warn] VQ init failed: {e}.")
            vq_model = vq_stride = None
    codebook_E = None
    if need_codebook:
        if vq_model is None:
            print("[Warn] Codebook-based features requested but no VQ provided; disabling them.")
            use_emb_consistency = False
            use_emb_grad_ce = False
            neighbor_idx = None
        else:
            with torch.no_grad():
                tok_all = torch.arange(vocab, device=device).reshape(1, -1)
                vq_raw = unwrap_compiled(vq_model)
                codebook_E = vq_raw.codebook(tok_all).squeeze(0).detach()
            if (neighbor_idx is not None) and callable(neighbor_idx):
                E = F.normalize(codebook_E, dim=-1)
                sim = (E @ E.t())
                sim.fill_diagonal_(-1.0)
                k = int(max(1, neighbor_k))
                neighbor_idx = sim.topk(k=k, dim=-1).indices.detach().cpu().numpy()  # numpy int array [V,k]
    ce_weight = None
    if (cf_counts is not None) and (cf_beta > 0.0):
        w = torch.tensor(cf_counts, dtype=torch.float32, device=device)
        w = torch.pow(torch.clamp(w, min=1.0), -cf_beta)
        w = w / w.mean()
        ce_weight = w
    if use_mtp:
        if int(mtp_horizons) < 2:
            raise ValueError(f"mtp_horizons must be >=2 when use_mtp=True, got {mtp_horizons}")
        if mtp_weight == "geom":
            ws = torch.tensor([mtp_geom_gamma ** (k-1) for k in range(1, mtp_horizons+1)], dtype=torch.float32, device=device)
            ws = ws / ws.sum()
        else:
            ws = torch.ones(mtp_horizons, dtype=torch.float32, device=device) / float(mtp_horizons)
    else:
        ws = None
    def epoch_pass(dl, train=True, use_mtp_flag=False):
        if not train and use_ema and ema is not None and ema_eval_only: ema.apply_shadow(model)
        model.train(train)
        total_loss=0.0; total_tok=0; acc1_sum=0.0
        acc5_list=[]; acc10_list=[]; ece_list=[]
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            if id_offset_session: batch["session_idB"] = batch["session_idB"] + int(id_offset_session)
            if id_offset_neuron: batch["neuron_idsBN"] = batch["neuron_idsBN"] + int(id_offset_neuron)
            batch_pack = build_ar_batch(batch, cfg, training=train)
            if batch_pack is None: raise RuntimeError("build_ar_batch returned None (unexpected in DDP-safe version).")
            tokens_inBNT, targetsBNT, neuron_idsBNT, time_pos_inBT, time_kpmask_inBT, target_validBNT, session_idB = batch_pack
            tokens_in_clean = tokens_inBNT
            tokens_in_used = tokens_in_clean
            if train and (tok_noise_p > 0.0) and (isinstance(neighbor_idx, np.ndarray)):
                B_, Nsel, T1 = tokens_in_used.shape
                mask = (torch.rand((B_, Nsel, T1), device=device) < tok_noise_p)
                v_src = tokens_in_used[mask]
                if v_src.numel() > 0:
                    nb = neighbor_idx[v_src.detach().cpu().numpy()]  # [M,k]
                    rnd = np.random.randint(0, nb.shape[1], size=(nb.shape[0],))
                    v_dst = torch.from_numpy(nb[np.arange(nb.shape[0]), rnd]).to(device)
                    tokens_in_used = tokens_in_used.clone()
                    tokens_in_used[mask] = v_dst
            use_ss_now = (train and use_scheduled_sampling and ss_prob > 0.0 and tokens_in_used.size(2) > 1)
            logits_teacher = None
            if use_ss_now and (random.random() < ss_prob):
                with torch.no_grad():
                    logits_teacher = model(tokens_in_used, neuron_idsBNT, time_pos_inBT, time_kpmask_inBT, session_idB)
                    pred_all = logits_teacher.argmax(dim=-1)
                pred_prev = torch.cat([tokens_in_used[:, :, :1], pred_all[:, :, :-1]], dim=2)
                B_, _, T1 = tokens_in_used.shape
                L = min(ss_block_len, T1)
                starts = torch.randint(low=0, high=T1-L+1, size=(B_,), device=device)
                mask = torch.zeros((B_, 1, T1), dtype=torch.bool, device=device)
                for b in range(B_): s = int(starts[b].item()); mask[b, 0, s:s+L] = True
                mask = mask.expand(B_, tokens_in_used.size(1), T1)
                tokens_in_used = torch.where(mask, pred_prev, tokens_in_used)
            with _autocast_ctx(device=device, use_bf16=use_bf16):
                if use_mtp_flag:
                    logits, hid_base = model(tokens_in_used, neuron_idsBNT, time_pos_inBT, time_kpmask_inBT, session_idB=session_idB, return_h=True)
                else:
                    logits = model(tokens_in_used, neuron_idsBNT, time_pos_inBT, time_kpmask_inBT, session_idB=session_idB, return_h=False)
                    hid_base = None
                logp   = F.log_softmax(logits, dim=-1)
                valid  = target_validBNT
                n_valid = int(valid.sum().item())
                has_valid = (n_valid > 0)
                if not has_valid:
                    loss = logits.sum() * 0.0
                    acc1 = torch.zeros((), device=device)
                else:
                    logp_flat = logp[valid]
                    tgt_flat  = targetsBNT[valid]
                    if use_emb_grad_ce and (codebook_E is not None):
                        Ein = tokens_in_used[valid]
                        E   = codebook_E
                        E_tgt = E[tgt_flat]
                        E_in  = E[Ein]
                        w_local = torch.linalg.vector_norm(E_tgt - E_in, dim=-1)
                        mean_w  = (w_local.mean().detach() + 1e-8)
                        w_local = torch.pow(torch.clamp(w_local / mean_w, min=emb_grad_floor), emb_grad_alpha)
                        nll = F.nll_loss(logp_flat, tgt_flat, reduction='none', weight=ce_weight)
                        loss_h1 = (nll * w_local).mean()
                    else: loss_h1 = F.nll_loss(logp_flat, tgt_flat, reduction='mean', weight=ce_weight)
                    loss_ec_h1 = .0
                    if use_emb_consistency and (codebook_E is not None):
                        p_flat = logp_flat.exp()
                        E = codebook_E
                        E_pred = p_flat @ E
                        E_gt   = E[tgt_flat]
                        loss_ec_h1 = F.mse_loss(E_pred, E_gt, reduction='mean') * float(emb_consis_lambda)
                    kd_loss = .0
                    if train and use_ngram_kd and hasattr(epoch_pass, "_ngram_counts"):
                        n = int(ngram_order)
                        lap = float(ngram_laplace)
                        last_valid = target_validBNT[:, :, -1]          # [B_, Nsel] bool
                        if last_valid.any():
                            prev = tokens_in_used[:, :, -n:]            # [B_, Nsel, n]
                            prev = prev[last_valid]                     # [M, n]
                            p = logits[:, :, -1, :][last_valid].softmax(dim=-1)  # [M, V]
                            M, V = p.shape
                            prev_np = prev.detach().cpu().numpy()       # [M, n]
                            keys = [prev_np[i].tobytes() for i in range(M)]
                            groups = defaultdict(list)
                            for i_k, kb in enumerate(keys): groups[kb].append(i_k)
                            q_list = []
                            p_list = []
                            ngram_dict = epoch_pass._ngram_counts  # type: ignore
                            for kb, idxs in groups.items():
                                one_prev = np.frombuffer(kb, dtype=prev_np.dtype, count=n)
                                tup = tuple(int(x) for x in one_prev.tolist())
                                cnt = ngram_dict.get(tup, None)
                                if cnt is None: continue
                                cnt_t = torch.from_numpy(cnt.astype(np.float32)).to(device)  # [V]
                                q = (cnt_t + lap) / (cnt_t.sum() + lap * float(V))           # [V]
                                ii = torch.tensor(idxs, device=device, dtype=torch.long)
                                p_g = p.index_select(0, ii)                                  # [g, V]
                                q_g = q.unsqueeze(0).expand(p_g.size(0), V)                  # [g, V]
                                p_list.append(p_g)
                                q_list.append(q_g)
                            if p_list:
                                P = torch.cat(p_list, dim=0).clamp_min(1e-9)
                                Q = torch.cat(q_list, dim=0).clamp_min(1e-9)
                                kd = (P * (P.log() - Q.log())).sum(dim=-1).mean()
                                kd_loss = float(ngram_alpha) * kd
                    if use_mtp_flag and (ws is not None):
                        B_, Nsel, T1 = tokens_in_used.shape
                        x_full = torch.cat([tokens_in_clean, targetsBNT[:, :, -1:].detach()], dim=2)  # [B,Nsel,T_in+1]
                        Tw = x_full.size(2)
                        time_kpmask_full = torch.zeros((B_, Tw), dtype=time_kpmask_inBT.dtype, device=device)
                        time_kpmask_full[:, :-1] = time_kpmask_inBT
                        time_kpmask_full[:, -1]  = target_validBNT[:, :, -1].any(dim=1)
                        ws2 = ws[1:]
                        ws2 = ws2 / ws2.sum()
                        loss_total = 0.0
                        ec_total   = 0.0
                        for k in range(2, mtp_horizons+1):
                            Lk = Tw - k               
                            if Lk <= 0: continue
                            hid_k  = hid_base[:, :, :Lk, :]                     # [B,Nsel,Lk,D]
                            tars_k = x_full[:, :,  k:   ]                       # [B,Nsel,Lk]
                            valid_k_BNT = time_kpmask_full[:, k:].unsqueeze(1).expand(B_, Nsel, Lk)
                            if int(valid_k_BNT.sum()) == 0: continue
                            logits_k = unwrap_compiled(model).mtp_heads[k-2](hid_k)
                            logp_k   = F.log_softmax(logits_k, dim=-1)
                            loss_k = masked_ce(logits_k, tars_k, valid_k_BNT, weight=ce_weight)
                            if use_emb_consistency and emb_consis_apply_to_mtp and (codebook_E is not None):
                                p_k_flat   = logits_k[valid_k_BNT].softmax(dim=-1)
                                E_pred_k   = p_k_flat @ codebook_E
                                E_gt_k     = codebook_E[tars_k[valid_k_BNT]]
                                ec_k       = F.mse_loss(E_pred_k, E_gt_k, reduction='mean') * float(emb_consis_lambda)
                                ec_total   = ec_total + ws2[k-2] * ec_k
                            loss_total = loss_total + ws2[k-2] * loss_k
                        loss = loss_h1 + loss_ec_h1 + kd_loss + float(mtp_lambda) * loss_total + ec_total
                    else: loss = loss_h1 + loss_ec_h1 + kd_loss
                    base_logits_for_metric = logits
                    logits_flat_m = base_logits_for_metric[valid]
                    tgt_flat_m    = tgt_flat
                    pred = logits_flat_m.argmax(dim=-1)
                    acc1 = (pred == tgt_flat_m).float().mean()
            if train:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step(); scheduler.step()
                if use_ema and ema is not None: ema.update(model)
            if has_valid:
                total_loss += float(loss.item()) * n_valid
                total_tok  += n_valid
                acc1_sum   += float(acc1) * n_valid
                accs = topk_acc_from_logits(logits_flat_m, tgt_flat_m, ks=(5,10))
                acc5_list.append(accs['acc@5']); acc10_list.append(accs['acc@10'])
                ece_list.append(ece_score(logits_flat_m, tgt_flat_m, n_bins=15))
        if not train and use_ema and ema is not None and ema_eval_only: ema.restore(model)
        if total_tok == 0:
            out = dict(loss=float('nan'), ppl=float('inf'), acc1=0.0, acc5=0.0, acc10=0.0, ece=float('nan'))
            return out
        acc5_sum = float(sum(acc5_list))
        acc5_cnt = float(len(acc5_list))
        acc10_sum = float(sum(acc10_list))
        acc10_cnt = float(len(acc10_list))
        ece_sum = float(sum(ece_list))
        ece_cnt = float(len(ece_list))
        if _dist_avail_and_initialized():
            t = torch.tensor(
                [
                    float(total_loss),
                    float(total_tok),
                    float(acc1_sum),
                    float(acc5_sum),
                    float(acc5_cnt),
                    float(acc10_sum),
                    float(acc10_cnt),
                    float(ece_sum),
                    float(ece_cnt),
                ],
                device=device if (isinstance(device, str) and device.startswith('cuda')) else 'cpu',
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, total_tok, acc1_sum, acc5_sum, acc5_cnt, acc10_sum, acc10_cnt, ece_sum, ece_cnt = [x.item() for x in t]
        loss_mean = float(total_loss / max(1.0, total_tok))
        out = dict(
            loss=loss_mean,
            ppl=math.exp(loss_mean),
            acc1=float(acc1_sum / max(1.0, total_tok)),
            acc5=float(acc5_sum / max(1.0, acc5_cnt)) if acc5_cnt > 0 else 0.0,
            acc10=float(acc10_sum / max(1.0, acc10_cnt)) if acc10_cnt > 0 else 0.0,
            ece=float(ece_sum / max(1.0, ece_cnt)) if ece_cnt > 0 else float('nan'),
        )
        return out
    if use_ngram_kd and hasattr(run_axial_ar, "_ngram_counts"): epoch_pass._ngram_counts = run_axial_ar._ngram_counts  # type: ignore
    out_path_greedy = None
    out_path_beam = None
    tf_out_path = None
    for ep in range(1, epochs+1):
        if _dist_avail_and_initialized() and hasattr(dl_tr, 'sampler') and hasattr(dl_tr.sampler, 'set_epoch'): dl_tr.sampler.set_epoch(ep)
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        use_mtp_flag = use_mtp and (getattr(unwrap_compiled(model), "mtp_heads", None) is not None)
        timer = EpochTimer(device=device)
        timer.start()
        tr = epoch_pass(dl_tr, train=True,  use_mtp_flag=use_mtp_flag and (ep > mtp_warmup_epochs))
        t_train = timer.stop()
        timer.start()
        va = epoch_pass(dl_va, train=False, use_mtp_flag=use_mtp_flag and (ep > mtp_warmup_epochs))
        t_val = timer.stop()
        mem_gb = None
        time_str = f" | time: train={t_train:.2f}s val={t_val:.2f}s"
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available(): mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        cur_lr = scheduler.get_last_lr()[0]
        ema_tag = " EMA" if (use_ema and ema is not None and ema_eval_only) else ""
        if mem_gb is not None: time_str += f" | peak_mem={mem_gb:.2f}GB"
        log_print(
            f"[Axial-AR{ema_tag}] EP{ep:02d} | lr={cur_lr:.6g} | "
            f"train: ppl={tr['ppl']:.2f} loss={tr['loss']:.3f} acc1={tr['acc1']:.3f}  "
            f"| val: ppl={va['ppl']:.2f} loss={va['loss']:.3f} acc1={va['acc1']:.3f} "
            f"acc5={va['acc5']:.3f} acc10={va['acc10']:.3f} ECE={va['ece']:.3f}"
            + time_str, log_txt_path=log_txt_path
        )
        if (dl_va is not None) and (meta is not None) and (vq_model is not None) and (vq_stride is not None):
            if (ep % max(1, int(eval_detail_every)) == 0):
                use_ema_flag = (use_ema and ema is not None and ema_eval_only)
                if _dist_avail_and_initialized():
                    dist.barrier()
                eval_multisession_val_vq_recon_rollout(
                    model=model,
                    dl_va=dl_va,
                    meta=meta,
                    vq_model=vq_model,
                    vq_stride=vq_stride,
                    prefix_tokens=int(gen_prefix),
                    device=device,
                    use_ema=use_ema_flag,
                    ema=ema,
                    id_offset_neuron=int(id_offset_neuron),
                    id_offset_session=int(id_offset_session),
                    print_per_trial=False,
                    log_txt_path=log_txt_path,
                )
                if _dist_avail_and_initialized(): dist.barrier()
        # if is_main_process() and (val_trials is not None) and ((ep % gen_every) == 0):
        #     os.makedirs(gen_outdir, exist_ok=True)
        #     out_path_greedy = os.path.join(gen_outdir, f"ar_gen_greedy_ep{ep:03d}.npz")
        #     out_path_beam = os.path.join(gen_outdir, f"ar_gen_beam_ep{ep:03d}.npz")
        #     use_ema_flag = (use_ema and ema is not None and ema_eval_only)
        #     tf_path = os.path.join(gen_outdir, f"ar_tf_ep{ep:03d}.npz")
        #     tf_out_path = ar_teacher_forced_generate_trials(
        #         model=model, trials=val_trials, device=device, prefix_len=gen_prefix,
        #         out_npz_path=tf_path, use_ema=use_ema_flag, ema=ema, temp=1.0, topk=0)
        #     print(f"[Gen-TF] epoch {ep}: → {tf_path} (teacher-forced; prefix={gen_prefix})")
        #     out_path_greedy = ar_greedy_generate_trials(
        #         model, val_trials, vocab, device, prefix_len=gen_prefix, out_npz_path=out_path_greedy,
        #         use_ema=use_ema_flag, ema=ema, temp=max(1e-6, decode_temp))
        #     if (vq_model is not None) and (gt_X is not None) and (vq_stride is not None):
        #         try:
        #             mean_corr_greedy = eval_ar_npz_mean_corr(out_path_greedy, gt_X, gt_L, vq_model, vq_stride, gen_prefix_tokens=gen_prefix, device=device)
        #             mean_corr_tf = eval_ar_npz_mean_corr(tf_out_path, gt_X, gt_L, vq_model, vq_stride, gen_prefix_tokens=gen_prefix, device=device)
        #         except Exception as e: print(f"[Warn] Corr evaluation failed at EP{ep}: {e}")
        #     ema_note = "; eval=EMA" if use_ema_flag else ""
        #     print(f"[Gen] epoch {ep}: → {out_path_greedy} (decode=greedy; prefix={gen_prefix}{ema_note})  mean_corr={mean_corr_greedy:.4f}")
        #     print(f"[Gen] epoch {ep}: → {tf_path} (decode=teacher forcing; prefix={gen_prefix}{ema_note})  mean_corr={mean_corr_tf:.4f}.")
        if is_main_process() and ckpt_every and (ep % int(ckpt_every) == 0):
            ckpt_path = os.path.join(ckpt_dir, f"axial_ar_ep{ep:03d}.pth")
            sd_raw = {k: v.detach().cpu() for k, v in model_raw.state_dict().items()}
            sd_ema = None
            if save_ema_weights and use_ema and (ema is not None):
                try:
                    ema.apply_shadow(model)
                    sd_ema = {k: v.detach().cpu() for k, v in model_raw.state_dict().items()}
                finally: ema.restore(model)
            ckpt = {
                "epoch": int(ep),
                "cfg": vars(model_raw.cfg), 
                "model": sd_raw,
                "model_ema": sd_ema, 
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "vocab": int(model_raw.cfg.vocab),
                "n_neurons": int(model_raw.cfg.n_neurons),
                "n_sessions": int(getattr(model_raw.cfg, "n_sessions", 0)),
                "id_offset_neuron": int(id_offset_neuron),
                "id_offset_session": int(id_offset_session),
                "session_paths": list(meta.get("npz_list", [])) if (meta is not None) else None,
                "sessions_meta": meta.get("sessions", None) if (meta is not None) else None,
            }
            torch.save(ckpt, ckpt_path)
            print(f"[CKPT] saved → {ckpt_path} (ema={'yes' if sd_ema is not None else 'no'})")
    return model

def _unwrap_object(v):
    if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1: return v.item()
    return v

def _stable_int_hash(s: str) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def load_npz_splits_tokens_v2(npz_path: str, token_key: str = "token", vocab: int = None):
    def _is_index_dict(d):
        if not isinstance(d, dict) or len(d) == 0: return False
        ks = list(d.keys())
        return all(str(k).isdigit() for k in ks)

    def _sort_keys(ks):
        return sorted(ks, key=lambda x: int(str(x)) if str(x).isdigit() else str(x))

    def _extract_tok(trial_obj, split: str):
        if isinstance(trial_obj, np.ndarray):
            if trial_obj.ndim >= 2: return trial_obj
            if trial_obj.shape == () and isinstance(trial_obj.item(), dict): return _extract_tok(trial_obj.item(), split)
        if isinstance(trial_obj, dict):
            if token_key in trial_obj: return trial_obj[token_key]
            for alt in ("tok", "tokens", "token_ids", "tokNT", "codes", "ids", "k"):
                if alt in trial_obj: return trial_obj[alt]
            raise KeyError(
                f"[load_npz_splits_tokens_v2] token_key='{token_key}' not found in trial dict "
                f"({npz_path} split={split}). keys={list(trial_obj.keys())[:30]}"
            )
        tok = getattr(trial_obj, token_key, None)
        if tok is not None: return tok
        arr = np.asarray(trial_obj)
        if arr.ndim >= 2: return arr
        raise ValueError(
            f"[load_npz_splits_tokens_v2] Cannot extract tokens from type={type(trial_obj)} "
            f"({npz_path} split={split})."
        )
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for split in ("train", "val", "test"):
            if split not in z: continue
            obj = z[split]
            trials = []
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                for item in obj:
                    if isinstance(item, dict) and (token_key not in item) and _is_index_dict(item):
                        for k in _sort_keys(item.keys()):
                            tok = _extract_tok(item[k], split)
                            trials.append(np.asarray(tok, dtype=np.int64))
                    else:
                        tok = _extract_tok(item, split)
                        trials.append(np.asarray(tok, dtype=np.int64))
            elif isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
                d = obj.item()
                if (token_key not in d) and _is_index_dict(d):
                    for k in _sort_keys(d.keys()):
                        tok = _extract_tok(d[k], split)
                        trials.append(np.asarray(tok, dtype=np.int64))
                else:
                    tok = _extract_tok(d, split)
                    trials.append(np.asarray(tok, dtype=np.int64))
            else:
                tok = _extract_tok(obj, split)
                trials.append(np.asarray(tok, dtype=np.int64))
            if vocab is not None:
                for ti, t in enumerate(trials):
                    tmin, tmax = int(t.min()), int(t.max())
                    if tmin < 0 or tmax >= vocab:
                        raise ValueError(
                            f"[{npz_path} split={split}] token out of range at trial#{ti}: "
                            f"min={tmin} max={tmax} vocab={vocab}"
                        )
            out[split] = trials
    return out

def _update_unigram_counts(counts: np.ndarray, tokNT: np.ndarray):
    vals, cnts = np.unique(tokNT.reshape(-1), return_counts=True)
    counts[vals] += cnts

def compute_unigram_counts_from_sessions(meta, token_key="token", vocab: int = 128):
    counts = np.zeros(vocab, dtype=np.int64)
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):  _update_unigram_counts(counts, tokNT)
    return np.clip(counts, 1, None)

def compute_ngram_counts_from_sessions(meta, n: int, token_key="token", vocab: int = 128):
    assert n >= 1
    from collections import defaultdict
    ngram_counts = defaultdict(lambda: np.zeros(vocab, dtype=np.int64))
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):
            N, T = tokNT.shape
            if T <= n: continue
            for i in range(N):
                seq = tokNT[i]
                for t in range(n, T):
                    prev = tuple(int(x) for x in seq[t-n:t])
                    nxt = int(seq[t])
                    ngram_counts[prev][nxt] += 1
    return ngram_counts

def compute_bigram_logP_from_sessions(meta, token_key="token", vocab: int = 128, laplace: float = 0.5):
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    for s in meta["sessions"]:
        splits = load_npz_splits_tokens_v2(s["path"], token_key=token_key, vocab=vocab)
        for tokNT in splits.get("train", []):
            N, T = tokNT.shape
            if T < 2: continue
            prev = tokNT[:, :-1].reshape(-1)
            nxt  = tokNT[:, 1:].reshape(-1)
            for p, q in zip(prev, nxt): counts[int(p), int(q)] += 1
    probs = (counts + laplace) / (counts.sum(axis=1, keepdims=True) + laplace * vocab)
    return torch.from_numpy(np.log(probs).astype(np.float32))  # [V,V] logP

def load_ckpt_state_dict(ckpt_path: str, prefer_ema: bool = True):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and ("model" in raw):
        sd_raw = raw["model"]
        sd_ema = raw.get("model_ema", None)
        saved_cfg = raw.get("cfg", None)
    else:
        sd_raw = raw
        sd_ema = None
        saved_cfg = None
    sd_raw = normalize_state_dict_keys(sd_raw)
    sd_ema = normalize_state_dict_keys(sd_ema) if isinstance(sd_ema, dict) else None
    sd_use = sd_ema if (prefer_ema and sd_ema is not None) else sd_raw
    return raw, sd_use, sd_raw, sd_ema, saved_cfg

def main():
    raise RuntimeError(
        "Legacy standalone CLI entry has been disabled. Use task/dt_train_Tseng.py with Hydra/torchrun instead."
    )


if __name__ == "__main__":
    main()
