import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

def load_session_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    tids = set()
    for k in data.files:
        if k.startswith("trial_") and "_gt_trace" in k:
            # format: trial_{tid}_gt_trace
            parts = k.split("_")
            # trial, {tid}, gt, trace
            if parts[1].isdigit():
                tids.add(int(parts[1]))
    
    sorted_tids = sorted(list(tids))
    trials = []
    sid = -1
    
    for tid in sorted_tids:
        gt_key = f"trial_{tid}_gt_trace"
        pr_key = f"trial_{tid}_pred_trace"
        
        if gt_key not in data or pr_key not in data:
            continue
            
        gt = data[gt_key]   # [N, T]
        pr = data[pr_key]   # [N, T]
        
        gt_T = gt.T
        pr_T = pr.T
        
        trials.append((gt_T, pr_T))
        
        if f"trial_{tid}_session_id" in data:
            sid = int(data[f"trial_{tid}_session_id"])

    if sid == -1:
        basename = os.path.basename(npz_path)
        try:
            sid = int(basename.replace("pred_session_", "").replace(".npz", ""))
        except:
            sid = 0
            
    return sid, trials

def analyze_session_pca(sid, trials, n_components, save_dir, norm_mode='none', skip_frames=4, eval_len=-1, shuffle=False):
    os.makedirs(save_dir, exist_ok=True)
    
    valid_gt = []
    valid_pred = []
    
    for gt, pred in trials:
        if skip_frames > 0:
            if len(gt) <= skip_frames: continue
            gt = gt[skip_frames:]
            pred = pred[skip_frames:]
        
        if eval_len > 0:
            if len(gt) < eval_len: continue
            gt = gt[:eval_len]
            pred = pred[:eval_len]

        if norm_mode == 'demean':
            gt = gt - gt.mean(axis=0, keepdims=True)
            pred = pred - pred.mean(axis=0, keepdims=True)
        elif norm_mode == 'zscore':
            gt_mean = gt.mean(axis=0, keepdims=True)
            gt_std = gt.std(axis=0, keepdims=True) + 1e-9
            gt = (gt - gt_mean) / gt_std
            
            pred_mean = pred.mean(axis=0, keepdims=True)
            pred_std = pred.std(axis=0, keepdims=True) + 1e-9
            pred = (pred - pred_mean) / pred_std
            
        valid_gt.append(gt)
        valid_pred.append(pred)
        
    if not valid_gt:
        return None, None

    if shuffle:
        import numpy as np
        perm_idx = np.random.permutation(len(valid_pred))
        valid_pred = [valid_pred[i] for i in perm_idx]

    X_fit = np.concatenate(valid_gt, axis=0) # [Total_T, N]
    pca = PCA(n_components=n_components)
    pca.fit(X_fit)
    
    expl_var = pca.explained_variance_ratio_
    
    pc_corrs = {k: [] for k in range(n_components)}
    
    z_gt_list = []
    z_pred_list = []
    
    for gt, pred in zip(valid_gt, valid_pred):
        z_gt = pca.transform(gt)
        z_pred = pca.transform(pred)
        
        z_gt_list.append(z_gt)
        z_pred_list.append(z_pred)
        
        min_len = min(len(z_gt), len(z_pred))

        for k in range(n_components):
            c_gt = z_gt[:min_len, k]
            c_pred = z_pred[:min_len, k]
            
            if min_len < 3 or np.std(c_gt) < 1e-9 or np.std(c_pred) < 1e-9:
                c = 0.0
            else:
                c, _ = pearsonr(c_gt, c_pred)
            pc_corrs[k].append(c)
            
    metrics = {
        "session_id": sid,
        "n_trials": len(trials),
        "total_expl_var_gt": np.sum(expl_var),
    }
    for k in range(n_components):
        metrics[f"PC{k}_corr_mean"] = np.mean(pc_corrs[k])
        metrics[f"PC{k}_corr_std"] = np.std(pc_corrs[k])
        metrics[f"PC{k}_expl_var"] = expl_var[k]
        
    metrics["Top3_corr_mean"] = np.mean([metrics[f"PC{k}_corr_mean"] for k in range(min(3, n_components))])

    best_idx = np.argmax([np.mean([pc_corrs[k][i] for k in range(min(3, n_components))]) for i in range(len(valid_gt))])
    
    z_gt_best = z_gt_list[best_idx]
    z_pred_best = z_pred_list[best_idx]
    
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))
    # PC0
    axs[0].plot(z_gt_best[:, 0], 'k', label='GT')
    axs[0].plot(z_pred_best[:, 0], 'r--', label='Pred')
    axs[0].set_title(f"PC0 (r={pc_corrs[0][best_idx]:.2f})")
    
    # PC1
    axs[1].plot(z_gt_best[:, 1], 'k')
    axs[1].plot(z_pred_best[:, 1], 'r--')
    axs[1].set_title(f"PC1 (r={pc_corrs[1][best_idx]:.2f})")
    
    # PC2
    axs[2].plot(z_gt_best[:, 2], 'k')
    axs[2].plot(z_pred_best[:, 2], 'r--')
    axs[2].set_title(f"PC2 (r={pc_corrs[2][best_idx]:.2f})")
    
    # 2D Traj
    axs[3].plot(z_gt_best[:, 0], z_gt_best[:, 1], 'k', alpha=0.6, label='GT')
    axs[3].plot(z_pred_best[:, 0], z_pred_best[:, 1], 'r--', alpha=0.6, label='Pred')

    if len(z_gt_best) > 1:
        axs[3].plot(z_gt_best[0, 0], z_gt_best[0, 1], 'ko', markersize=5, label='Start') # Start
        axs[3].arrow(z_gt_best[-2, 0], z_gt_best[-2, 1], 
                     z_gt_best[-1, 0] - z_gt_best[-2, 0], 
                     z_gt_best[-1, 1] - z_gt_best[-2, 1], 
                     shape='full', lw=0, length_includes_head=True, head_width=0.05, color='k')

    if len(z_pred_best) > 1:
        axs[3].plot(z_pred_best[0, 0], z_pred_best[0, 1], 'ro', markersize=5) # Start
        axs[3].arrow(z_pred_best[-2, 0], z_pred_best[-2, 1], 
                     z_pred_best[-1, 0] - z_pred_best[-2, 0], 
                     z_pred_best[-1, 1] - z_pred_best[-2, 1], 
                     shape='full', lw=0, length_includes_head=True, head_width=0.05, color='r')

    axs[3].set_title(f"2D Traj (S{sid} T{best_idx})")
    axs[3].legend()
    
    pc_means = [metrics[f"PC{k}_corr_mean"] for k in range(n_components)]
    pc_stds = [metrics[f"PC{k}_corr_std"] for k in range(n_components)]
    
    n_plot = n_components
    xs = np.arange(n_plot)
    ys = np.array(pc_means[:n_plot])
    errs = np.array(pc_stds[:n_plot])
    
    axs[4].plot(xs, ys, 'b-o', label='Mean Corr')
    axs[4].fill_between(xs, ys - errs, ys + errs, color='b', alpha=0.2, label='Std')
    axs[4].set_title(f"Top {n_plot} PC Corr (Mean+Std)")
    axs[4].set_xlabel("PC Index")
    axs[4].set_ylabel("Correlation")
    axs[4].set_xticks(xs)
    axs[4].set_ylim(-0.2, 1.1)
    axs[4].grid(True, alpha=0.3)
    axs[4].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"session_{sid}_pca_traj.png"))
    plt.close()
    
    return metrics, pc_corrs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default=None, help="Directory containing raw & forecasting trace.npz")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--n_components", type=int, default=20)
    parser.add_argument("--norm_mode", type=str, default="none", choices=["none", "demean", "zscore"], help="Normalization: none, demean, or zscore")
    parser.add_argument("--demean", type=int, default=0, help="Deprecated: use norm_mode. If 1, sets norm_mode='demean'")
    parser.add_argument("--skip_frames", type=int, default=40, help="Skip 4 frames: Gen(36)+Ana(4)=40 to match Old(40).")
    parser.add_argument("--eval_len", type=int, default=-1, help="Evaluation length after skip. -1 uses all. Short trials discarded.")
    parser.add_argument("--session_range", type=int, nargs=2, default=None, help="Only process sessions with IDs in [start, end] inclusive.")
    parser.add_argument("--shuffle", action="store_true", help="If set, shuffle trials within session to compute null distribution.")
    args = parser.parse_args()

    args.pred_dir = os.path.expanduser(args.pred_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.demean == 1 and args.norm_mode == "none":
        args.norm_mode = "demean"

    files = sorted(glob.glob(os.path.join(args.pred_dir, "pred_session_*.npz")))
    
    if args.session_range is not None:
        start_id, end_id = args.session_range
        filtered_files = []
        for f in files:
            try:
                basename = os.path.basename(f)
                sid_part = basename.replace("pred_session_", "").replace(".npz", "")
                sid = int(sid_part)
                if start_id <= sid <= end_id:
                    filtered_files.append(f)
            except ValueError:
                continue
        files = filtered_files
        print(f"Filtered files to range [{start_id}, {end_id}]. Count: {len(files)}")

    if not files:
        print(f"No files found in {args.pred_dir} (after filtering options)")
        return

    all_metrics = []
    pooled_corrs = {k: [] for k in range(args.n_components)}
    
    print(f"Found {len(files)} session files. Mode: {args.norm_mode}")
    for f in files:
        sid, trials = load_session_data(f)
        print(f"Processing Session {sid} ({len(trials)} trials)...")
        
        m, raw_corrs = analyze_session_pca(sid, trials, args.n_components, args.save_dir, norm_mode=args.norm_mode, skip_frames=args.skip_frames, eval_len=args.eval_len, shuffle=args.shuffle)
        if m:
            all_metrics.append(m)
            for k in raw_corrs:
                pooled_corrs[k].extend(raw_corrs[k])

            print(f"  [S{sid}] Top3 Corr: {m['Top3_corr_mean']:.3f}, EV_GT: {m['total_expl_var_gt']:.3f}")
            print(f"  Top PC Correlations (Mean +/- Std):")
            for k in range(args.n_components):
                print(f"    PC{k}: {m[f'PC{k}_corr_mean']:.4f} +/- {m[f'PC{k}_corr_std']:.4f}")

    if not all_metrics:
        print("No valid metrics computed.")
        return

    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(args.save_dir, "pca_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics summary to {csv_path}")
    
    print("\n=== Global Analysis (Inter-Session Statistics) ===")
    n_plot = args.n_components
    pca_cols = [f"PC{k}_corr_mean" for k in range(n_plot)]
    
    valid_cols = [c for c in pca_cols if c in df.columns]
    
    if valid_cols:
        means = df[valid_cols].mean()
        stds = df[valid_cols].std()
        
        print(f"Top {len(valid_cols)} PC Correlations (Session-Averaged Mean +/- Std across {len(df)} sessions):")
        for c in valid_cols:
            k = c.split("_")[0] # PC0, PC1...
            print(f"  {k}: {means[c]:.4f} +/- {stds[c]:.4f}")

    print("\n=== Global Analysis (Pooled Trial Statistics) ===")
    print(f"Top {args.n_components} PC Correlations (Pooled Mean +/- Std across all {sum(len(v) for v in pooled_corrs.values()) // args.n_components if args.n_components > 0 else 0} trials):")
    for k in range(args.n_components):
        if pooled_corrs[k]:
            mu = np.mean(pooled_corrs[k])
            sigma = np.std(pooled_corrs[k])
            print(f"  PC{k}: {mu:.4f} +/- {sigma:.4f}")

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if "session_id" in df.columns:
        sns.barplot(data=df, x="session_id", y="Top3_corr_mean", color="skyblue")
        plt.title("Mean Correlation of Top 3 PCs per Session")
        plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if valid_cols:
        xs = np.arange(len(valid_cols))
        ys = means.values
        errs = stds.values
        
        plt.plot(xs, ys, 'b-o', label='Session Mean')
        plt.fill_between(xs, ys - errs, ys + errs, color='b', alpha=0.2, label='Session Std')
        
        plt.title(f"PC Correlation across {len(df)} Sessions")
        plt.xlabel("PC Index")
        plt.ylabel("Correlation")
        plt.xticks(xs, [c.split('_')[0] for c in valid_cols])
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "pca_summary_plot.png"))
    print(f"Saved summary plot to {os.path.join(args.save_dir, 'pca_summary_plot.png')}")

if __name__ == "__main__":
    main()
