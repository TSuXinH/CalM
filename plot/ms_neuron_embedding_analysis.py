import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import pearsonr
import scipy.io
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding/tuning analysis")
    parser.add_argument("--data_npz", type=str, default=None, help="Path to data containing neural embeddings and tuning values (npz format)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for plots and cached outputs")
    parser.add_argument("--subs", nargs="*", default=["sub-6","sub-10"], help="Optional list of subject IDs to include (e.g. sub-3 sub-4). Omit for all")
    parser.add_argument("--layer", type=str, default="all", choices=["L23", "L5", "all"], help="Layer filter applied to session names")
    parser.add_argument("--include_unprocessed", action="store_true", help="Include sessions without tuning values (NaNs)")
    parser.add_argument("--pca_threshold", type=float, default=0.5, help="|d'| threshold for PCA strong-neuron subset")
    parser.add_argument("--lda_threshold", type=float, default=0.0, help="|d'| threshold for selecting neurons to fit LDA")
    parser.add_argument("--region_palette", type=str, default="tab20", help="Matplotlib colormap for region-colored PCA plot")
    parser.add_argument("--shuffle_seed", type=int, default=2026, help="Seed for LDA label shuffling")
    return parser.parse_args()


def load_analysis_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path, allow_pickle=True)
    required_keys = ["neu_emb", "cue_tuning", "choice_tuning", "session_names", "session_counts", "session_offsets", "session_processed"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}")
    return {k: data[k] for k in required_keys}


def parse_session_tags(session_name):
    subject = ""
    layer = ""
    region = ""

    subj_match = re.search(r"(sub-\d+)", session_name)
    if subj_match:
        subject = subj_match.group(1)

    layer_match = re.search(r"-(L\d+)", session_name)
    if layer_match:
        layer = layer_match.group(1)

    area_match = re.search(r"area-([A-Za-z0-9]+)", session_name)
    if area_match:
        region = area_match.group(1).upper()

    return subject, layer, region


def build_session_table(meta):
    sessions = []
    names = meta["session_names"].astype(str)
    counts = meta["session_counts"].astype(int)
    offsets = meta["session_offsets"].astype(int)
    processed_flags = meta["session_processed"].astype(bool)
    for name, count, offset, processed in zip(names, counts, offsets, processed_flags):
        subject, layer, region = parse_session_tags(name)
        sessions.append({
            "name": name,
            "subject": subject,
            "layer": layer,
            "region": region,
            "count": int(count),
            "offset": int(offset),
            "processed": bool(processed),
        })
    return sessions


def filter_sessions(sessions, subs, layer, include_unprocessed):
    selected = []
    subs_set = set(subs) if subs else None
    for sess in sessions:
        if subs_set and sess["subject"] not in subs_set:
            continue
        if layer != "all" and sess["layer"] != layer:
            continue
        if not include_unprocessed and not sess["processed"]:
            continue
        selected.append(sess)
    return selected


def build_masks(meta, sessions_selected):
    total_neurons = meta["cue_tuning"].shape[0]
    mask = np.zeros(total_neurons, dtype=bool)
    regions = np.empty(total_neurons, dtype=object)
    regions[:] = ""
    for sess in sessions_selected:
        start = sess["offset"]
        end = start + sess["count"]
        mask[start:end] = True
        regions[start:end] = sess["region"]
    return mask, regions


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    print("Loading analysis_data.npz...")
    meta = load_analysis_npz(args.data_npz)
    sessions = build_session_table(meta)

    print("Filtering sessions...")
    sessions_filtered = filter_sessions(sessions, args.subs, args.layer, args.include_unprocessed)
    if not sessions_filtered:
        raise RuntimeError("No sessions satisfy the requested filters")

    mask_sessions, regions = build_masks(meta, sessions_filtered)
    neu_emb = meta["neu_emb"][mask_sessions]
    total_selected = neu_emb.shape[0]
    cue = meta["cue_tuning"][mask_sessions]
    choice = meta["choice_tuning"][mask_sessions]
    regions_sel = regions[mask_sessions]

    print(f"Total neurons selected: {total_selected}. Filtering for valid tuning...")
    valid_idx = (~np.isnan(cue)) & (~np.isnan(choice))
    if np.sum(valid_idx) < 100:
        raise RuntimeError("Not enough valid neurons after filtering; decrease filters or recompute data")

    X = neu_emb[valid_idx]
    y_cue = cue[valid_idx]
    y_choice = choice[valid_idx]
    region_valid = regions_sel[valid_idx]

    print(f"Valid neurons: {X.shape[0]}. Scaling data...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running Local PCA analysis...")
    fig = plt.figure(figsize=(28, 12))
    gs = fig.add_gridspec(2, 4)
    ax_pca_cue = fig.add_subplot(gs[0, 0])
    ax_pca_choice = fig.add_subplot(gs[0, 1])

    def plot_pca_subset(tuning_vals, title_prefix, axis, threshold):
        mask = np.abs(tuning_vals) > threshold
        axis.set_title(f"PCA | {title_prefix}\n|d'| > {threshold:.2f} (N={np.sum(mask)})")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        if np.sum(mask) < 10:
            axis.text(0.5, 0.5, "Insufficient neurons", ha="center")
            return
        X_sub = X_scaled[mask]
        vals_sub = tuning_vals[mask]
        pca_local = PCA(n_components=2)
        coords = pca_local.fit_transform(X_sub)
        vmax = np.percentile(np.abs(vals_sub), 98)
        scatter = axis.scatter(coords[:, 0], coords[:, 1], c=vals_sub, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=6, alpha=0.6)
        plt.colorbar(scatter, ax=axis, label=f"{title_prefix} d'")
        axis.text(0.02, 0.95, f"Var: {pca_local.explained_variance_ratio_[0]:.2f}/{pca_local.explained_variance_ratio_[1]:.2f}", transform=axis.transAxes)

    plot_pca_subset(y_cue, "Cue", ax_pca_cue, args.pca_threshold)
    plot_pca_subset(y_choice, "Choice", ax_pca_choice, args.pca_threshold)

    print("Running Global PCA (Region Colored)...")
    ax_pca_region = fig.add_subplot(gs[0, 2])
    pca_all = PCA(n_components=2)
    coords_all = pca_all.fit_transform(X_scaled)
    unique_regions = [r for r in np.unique(region_valid) if r]
    region_to_idx = {r: idx for idx, r in enumerate(unique_regions)}
    colors = plt.get_cmap(args.region_palette, max(1, len(region_to_idx)))
    region_indices = np.array([region_to_idx.get(r, -1) for r in region_valid])
    mask_known = region_indices >= 0
    sc = ax_pca_region.scatter(coords_all[mask_known, 0], coords_all[mask_known, 1], c=region_indices[mask_known], cmap=colors, s=4, alpha=0.5)
    ax_pca_region.set_title("PCA Colored by Region")
    ax_pca_region.set_xlabel("PC1")
    ax_pca_region.set_ylabel("PC2")
    if len(region_to_idx) > 0:
        ticks = np.arange(len(region_to_idx))
        cbar = plt.colorbar(sc, ax=ax_pca_region, ticks=ticks)
        labels = [r for r, _ in sorted(region_to_idx.items(), key=lambda kv: kv[1])]
        cbar.ax.set_yticklabels(labels)
    else:
        ax_pca_region.text(0.5, 0.5, "No region labels", ha="center")
        labels = []

    ax_region_counts = fig.add_subplot(gs[0, 3])
    counts = {r: int(np.sum(region_valid == r)) for r in region_to_idx.keys()}
    if counts:
        regions_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        lbls = [f"{r}\nN={n}" for r, n in regions_sorted]
        values = [n for _, n in regions_sorted]
        ax_region_counts.bar(range(len(values)), values, color="gray")
        ax_region_counts.set_xticks(range(len(lbls)))
        ax_region_counts.set_xticklabels(lbls, rotation=45, ha="right")
        ax_region_counts.set_ylabel("Neurons")
        ax_region_counts.set_title("Region Counts")
    else:
        ax_region_counts.text(0.5, 0.5, "No region info", ha="center")
        ax_region_counts.axis("off")

    print("Running LDA Analysis...")
    rng = np.random.default_rng(args.shuffle_seed)
    n_pca_components = min(50, X_scaled.shape[1], X_scaled.shape[0])
    pca_pre = PCA(n_components=n_pca_components)
    X_pca = pca_pre.fit_transform(X_scaled)

    lda_mask = (np.abs(y_cue) > args.lda_threshold) & (np.abs(y_choice) > args.lda_threshold)
    n_train = np.sum(lda_mask)
    lda_thresh_used = args.lda_threshold
    if n_train < 40:
        lda_mask = np.ones_like(lda_mask, dtype=bool)
        n_train = np.sum(lda_mask)
        lda_thresh_used = 0.0
        print("Warning: Not enough strong neurons for LDA; using all valid neurons")

    X_train = X_pca[lda_mask]
    y_cue_train = y_cue[lda_mask]
    y_choice_train = y_choice[lda_mask]

    labels = np.zeros(n_train, dtype=int)
    labels[(y_cue_train < 0) & (y_choice_train < 0)] = 0
    labels[(y_cue_train < 0) & (y_choice_train > 0)] = 1
    labels[(y_cue_train > 0) & (y_choice_train < 0)] = 2
    labels[(y_cue_train > 0) & (y_choice_train > 0)] = 3

    lda = LDA(n_components=2)
    lda.fit(X_train, labels)
    proj_all = lda.transform(X_pca)
    acc_train = lda.score(X_train, labels)

    lda_shuff = LDA(n_components=2)
    labels_shuff = rng.permutation(labels)
    lda_shuff.fit(X_train, labels_shuff)
    proj_shuff = lda_shuff.transform(X_pca)
    acc_shuff = lda_shuff.score(X_train, labels_shuff)

    def plot_proj(ax, proj, values, title, angle_deg=None):
        vmax = np.percentile(np.abs(values), 98)
        sca = ax.scatter(proj[:, 0], proj[:, 1], c=values, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=3, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        plt.colorbar(sca, ax=ax)
        
        if angle_deg is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            rad = np.radians(angle_deg)
            slope = np.tan(rad)
            
            x_vals = np.array(xlim)
            y_vals = slope * x_vals
            
            ax.plot(x_vals, y_vals, 'k--', linewidth=1.5, label=f"Axis {angle_deg:.1f}°")

            vec_len = (xlim[1] - xlim[0]) * 0.2
            ax.arrow(0, 0, vec_len * np.cos(rad), vec_len * np.sin(rad), 
                     head_width=vec_len*0.1, head_length=vec_len*0.15, fc='k', ec='k')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def get_opt_angle_deg(proj, target):
         reg = LinearRegression(fit_intercept=True).fit(proj, target)
         coef = reg.coef_ # [w1, w2]
         return np.degrees(np.arctan2(coef[1], coef[0]))

    deg_cue = get_opt_angle_deg(proj_all, y_cue)
    deg_choice = get_opt_angle_deg(proj_all, y_choice)
    deg_cue_s = get_opt_angle_deg(proj_shuff, y_cue)
    deg_choice_s = get_opt_angle_deg(proj_shuff, y_choice)

    ax_lda_cue = fig.add_subplot(gs[1, 0])
    plot_proj(ax_lda_cue, proj_all, y_cue, f"LDA True | Cue\nTrain |d'| > {lda_thresh_used:.2f}, Acc={acc_train:.2f}", angle_deg=deg_cue)

    ax_lda_choice = fig.add_subplot(gs[1, 1])
    plot_proj(ax_lda_choice, proj_all, y_choice, f"LDA True | Choice\nTrain |d'| > {lda_thresh_used:.2f}", angle_deg=deg_choice)

    ax_lda_shuff_cue = fig.add_subplot(gs[1, 2])
    plot_proj(ax_lda_shuff_cue, proj_shuff, y_cue, f"LDA Shuffled | Cue\nAcc={acc_shuff:.2f}", angle_deg=deg_cue_s)

    ax_lda_shuff_choice = fig.add_subplot(gs[1, 3])
    plot_proj(ax_lda_shuff_choice, proj_shuff, y_choice, "LDA Shuffled | Choice", angle_deg=deg_choice_s)

    def max_axis_corr(proj, target):
        if proj.shape[1] < 2: return 0.0
        reg = LinearRegression().fit(proj, target)
        pred = reg.predict(proj)
        return abs(pearsonr(pred, target)[0])

    c_cue = max_axis_corr(proj_all, y_cue)
    c_choice = max_axis_corr(proj_all, y_choice)
    c_cue_s = max_axis_corr(proj_shuff, y_cue)
    c_choice_s = max_axis_corr(proj_shuff, y_choice)


    fig.text(0.5, 0.04, f"MaxPlaneCorr | Cue={c_cue:.2f} ({deg_cue:.1f}°) Choice={c_choice:.2f} ({deg_choice:.1f}°) | Shuff C={c_cue_s:.2f} ({deg_cue_s:.1f}°) Ch={c_choice_s:.2f} ({deg_choice_s:.1f}°)", ha="center", fontsize=11)
    fig.suptitle(f"Post-hoc Analysis | Selected={total_selected} | Valid={X.shape[0]}", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    summary_path = os.path.join(args.out_dir, "posthoc_summary.png")
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    np.savez(os.path.join(args.out_dir, "posthoc_filtered_data.npz"),
             embeddings=X,
             cue_tuning=y_cue,
             choice_tuning=y_choice,
             regions=region_valid,
             selected_sessions=np.array([sess["name"] for sess in sessions_filtered], dtype=object),
             total_selected_neurons=int(total_selected))

    print(f"Selected sessions: {len(sessions_filtered)}")
    print(f"Valid neurons after masking: {X.shape[0]} / {total_selected}")
    print(f"Saved figure to {summary_path}")

    print("Exporting data for MATLAB...")

    mask_cue_sub = np.abs(y_cue) > args.pca_threshold
    pca_cue_coords = []
    pca_cue_vals = []
    if np.sum(mask_cue_sub) > 10:
        pca_cue_coords = PCA(n_components=2).fit_transform(X_scaled[mask_cue_sub])
        pca_cue_vals = y_cue[mask_cue_sub]

    mask_choice_sub = np.abs(y_choice) > args.pca_threshold
    pca_choice_coords = []
    pca_choice_vals = []
    if np.sum(mask_choice_sub) > 10:
        pca_choice_coords = PCA(n_components=2).fit_transform(X_scaled[mask_choice_sub])
        pca_choice_vals = y_choice[mask_choice_sub]

    mat_export = {
        'pca_coords_global': coords_all,
        'pca_region_idx': region_indices,
        'pca_region_names': labels if len(region_to_idx) > 0 else [],
        
        'pca_cue_coords': pca_cue_coords,
        'pca_cue_vals': pca_cue_vals,
        
        'pca_choice_coords': pca_choice_coords,
        'pca_choice_vals': pca_choice_vals,

        'lda_proj_all': proj_all,
        'lda_proj_shuff': proj_shuff,
        'y_cue': y_cue,
        'y_choice': y_choice,
        
        'acc_train': acc_train,
        'acc_shuff': acc_shuff,
        'corr_true_cue': c_cue,
        'corr_true_choice': c_choice,
        'corr_shuff_cue': c_cue_s,
        'corr_shuff_choice': c_choice_s,
        'lda_threshold': lda_thresh_used,
    }
    
    mat_path = os.path.join(args.out_dir, "posthoc_analysis_data.mat")
    scipy.io.savemat(mat_path, mat_export)
    print(f"Saved MATLAB data to {mat_path}")

    m_script = r"""% Reproduce Post-Hoc Embedding Analysis (Full)
clear; clc; close all;

% Load Data
data = load('posthoc_analysis_data.mat');

% Basic Info
fprintf('LDA Train Acc (True): %.4f\n', data.acc_train);
fprintf('LDA Train Acc (Shuff): %.4f\n', data.acc_shuff);

figure('Name', 'Post-Hoc Analysis (Full)', 'Color', 'w', 'Position', [50, 50, 1600, 800]);

% --- Row 1: PCA Analysis ---

% 1. Local PCA Cue
subplot(2, 4, 1);
if ~isempty(data.pca_cue_coords)
    scatter_plot_colored(data.pca_cue_coords, data.pca_cue_vals, 'Local PCA | Cue');
end

% 2. Local PCA Choice
subplot(2, 4, 2);
if ~isempty(data.pca_choice_coords)
    scatter_plot_colored(data.pca_choice_coords, data.pca_choice_vals, 'Local PCA | Choice');
end

% 3. Global PCA Region
subplot(2, 4, 3);
hold on;
try
    if isfield(data, 'pca_region_idx') && ~isempty(data.pca_region_idx)
        valid_mask = data.pca_region_idx >= 0;
        X = data.pca_coords_global(valid_mask, :);
        g = data.pca_region_idx(valid_mask, :); 
        
        % Use turbo or similar distinct map
        cmap_reg = turbo(max(g)+1);
        scatter(X(:,1), X(:,2), 4, g, 'filled', 'MarkerFaceAlpha', 0.5);
        colormap(gca, cmap_reg);
        title('Global PCA | Region');
        xlabel('PC1'); ylabel('PC2');
    end
catch
    text(0.5, 0.5, 'Region Plot Error', 'HorizontalAlignment', 'center');
end
axis square; grid on;

% 4. Regions Count (Placeholder)
subplot(2, 4, 4);
try
    h = histogram(data.pca_region_idx, 'Orientation', 'horizontal');
    title('Region Counts');
catch
end
axis off;

% --- Row 2: LDA Analysis ---

% 5. LDA (True) - Cue
subplot(2, 4, 5);
scatter_plot_colored(data.lda_proj_all, data.y_cue, ...
    ['LDA True | Cue (Corr: ' num2str(data.corr_true_cue, '%.3f') ')']);

% 6. LDA (True) - Choice
subplot(2, 4, 6);
scatter_plot_colored(data.lda_proj_all, data.y_choice, ...
    ['LDA True | Choice (Corr: ' num2str(data.corr_true_choice, '%.3f') ')']);

% 7. LDA (Shuff) - Cue
subplot(2, 4, 7);
scatter_plot_colored(data.lda_proj_shuff, data.y_cue, ...
    ['LDA Shuff | Cue (Corr: ' num2str(data.corr_shuff_cue, '%.3f') ')']);

% 8. LDA (Shuff) - Choice
subplot(2, 4, 8);
scatter_plot_colored(data.lda_proj_shuff, data.y_choice, ...
    ['LDA Shuff | Choice (Corr: ' num2str(data.corr_shuff_choice, '%.3f') ')']);

% Add Summary Text
check_exist = @(f) isfield(data, f);
txt_str = '';
if check_exist('lda_threshold'), txt_str = [txt_str sprintf('|d''| Thresh: %.2f  ', data.lda_threshold)]; end

annotation('textbox', [0.1, 0.01, 0.8, 0.04], 'String', txt_str, ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontSize', 10);


% --- Helper Function ---
function scatter_plot_colored(coords, vals, title_str)
    hold on;
    vmax = prctile(abs(vals), 98);
    scatter(coords(:,1), coords(:,2), 10, vals, 'filled', 'MarkerFaceAlpha', 0.4);
    colormap(gca, 'coolwarm');
    caxis([-vmax, vmax]);
    colorbar;
    title(title_str, 'Interpreter', 'none');
    xlabel('LD1'); ylabel('LD2');
    axis square; grid on;
end
"""
    m_script_path = os.path.join(args.out_dir, "plot_posthoc_analysis.m")
    with open(m_script_path, "w") as f:
        f.write(m_script)
    print(f"Saved MATLAB script to {m_script_path}")



if __name__ == "__main__":
    main()
