import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import numpy as np
import torch
torch.set_num_threads(4)
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

SESSION_PATHS = [
    # session_paths
]

@dataclass
class TrialData:
    trial_ids: np.ndarray
    lengths: np.ndarray
    neural_trials: List[np.ndarray]
    behavior_trials: List[np.ndarray]
    neural_dim: int
    behavior_dim: int

def process_arrays(padded_X, padded_Y, lengths, trial_ids) -> TrialData:
    padded_X = np.asarray(padded_X, dtype=np.float32)
    padded_Y = np.asarray(padded_Y, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)

    if padded_X.ndim != 3 or padded_Y.ndim != 3:
        raise ValueError("padded_X/padded_Y must be [trial, channel, time]")
    if padded_X.shape[0] != padded_Y.shape[0]:
        raise ValueError("padded_X/padded_Y trial counts mismatch")
    
    neural_dim = padded_X.shape[1]
    behavior_dim = padded_Y.shape[1]

    valid_mask = lengths > 0
    neural_trials: List[np.ndarray] = []
    behavior_trials: List[np.ndarray] = []
    valid_ids: List[int] = []
    valid_lengths: List[int] = []

    for idx in np.where(valid_mask)[0]:
        length = int(lengths[idx])
        neural_slice = padded_X[idx, :, :length].transpose(1, 0)
        behavior_slice = padded_Y[idx, :, :length].transpose(1, 0)
        neural_trials.append(neural_slice)
        behavior_trials.append(behavior_slice)
        valid_ids.append(int(trial_ids[idx]))
        valid_lengths.append(length)

    return TrialData(
        trial_ids=np.asarray(valid_ids, dtype=np.int64),
        lengths=np.asarray(valid_lengths, dtype=np.int32),
        neural_trials=neural_trials,
        behavior_trials=behavior_trials,
        neural_dim=neural_dim,
        behavior_dim=behavior_dim,
    )

def load_session_data(data_path: Path):
    print(f"Loading data from {data_path} ...")
    with np.load(data_path, allow_pickle=True) as npz:
        train_data = process_arrays(
            npz["train_padded_X"], npz["train_padded_Y"], npz["train_lengths"], npz["train_trial_ids"]
        )
        valid_data = process_arrays(
            npz["val_padded_X"], npz["val_padded_Y"], npz["val_lengths"], npz["val_trial_ids"]
        )
        test_data = process_arrays(
            npz["test_padded_X"], npz["test_padded_Y"], npz["test_lengths"], npz["test_trial_ids"]
        )
    return train_data, valid_data, test_data

def compute_corr(true_raw: np.ndarray, pred_raw: np.ndarray) -> np.ndarray:
    if true_raw.shape[0] == 0:
        return np.zeros(true_raw.shape[1], dtype=np.float64)
    corrs = np.full(true_raw.shape[1], np.nan, dtype=np.float64)
    for dim in range(true_raw.shape[1]):
        true_col = true_raw[:, dim]
        pred_col = pred_raw[:, dim]
        if true_col.std() == 0 or pred_col.std() == 0:
            continue
        corrs[dim] = np.corrcoef(true_col, pred_col)[0, 1]
    return corrs

def evaluate_predictions(Y_trues: List[np.ndarray], Y_preds: List[np.ndarray], label: str):
    test_metrics = {"corr": [], "r2": [], "mse": []}

    for y_true, y_pred in zip(Y_trues, Y_preds):
        test_metrics["corr"].append(compute_corr(y_true, y_pred))
        test_metrics["r2"].append(r2_score(y_true, y_pred, multioutput="raw_values"))
        test_metrics["mse"].append(mean_squared_error(y_true, y_pred, multioutput="raw_values"))

    mean_corr_per_dim = np.nanmean(np.array(test_metrics["corr"]), axis=0)
    mean_r2_per_dim = np.nanmean(np.array(test_metrics["r2"]), axis=0)
    mean_mse_per_dim = np.nanmean(np.array(test_metrics["mse"]), axis=0)

    avg_corr = np.nanmean(mean_corr_per_dim)
    avg_r2 = np.nanmean(mean_r2_per_dim)
    avg_mse = np.nanmean(mean_mse_per_dim)

    print("-" * 30)
    print(f"[{label}] Per-trial Test Results (Averaged across trials):")
    print(f"  Overall Avg Correlation: {avg_corr:.4f}")
    print(f"  Overall Avg R^2:         {avg_r2:.4f}")
    print(f"  Overall Avg MSE:         {avg_mse:.4f}")
    print("-" * 30)
    print("Per-dimension metrics (X, Y, Z...):")
    print(f"  Correlation: {np.array2string(mean_corr_per_dim, precision=4, floatmode='fixed')}")
    print(f"  R^2:         {np.array2string(mean_r2_per_dim, precision=4, floatmode='fixed')}")
    print(f"  MSE:         {np.array2string(mean_mse_per_dim, precision=4, floatmode='fixed')}")
    print("-" * 30)

    return {
        "avg_corr": avg_corr, "avg_r2": avg_r2, "avg_mse": avg_mse,
        "dim_corr": mean_corr_per_dim, "dim_r2": mean_r2_per_dim, "dim_mse": mean_mse_per_dim
    }


def create_lagged_features(neural_data: np.ndarray, lags: List[int]) -> np.ndarray:
    feature_list = []
    for lag in lags:
        if lag == 0:
            feature_list.append(neural_data)
        elif lag > 0:
            shifted = np.zeros_like(neural_data)
            shifted[lag:, :] = neural_data[:-lag, :]
            feature_list.append(shifted)
        else:
            abs_lag = abs(lag)
            shifted = np.zeros_like(neural_data)
            shifted[:-abs_lag, :] = neural_data[abs_lag:, :]
            feature_list.append(shifted)
    return np.concatenate(feature_list, axis=1)

def prepare_design_matrix(data: TrialData, lags: List[int], scaler: StandardScaler = None, fit_scaler: bool = False):
    max_lag_future = max(0, -min(lags)) if lags else 0
    X_parts = []
    Y_parts = []
    for neural, behavior in zip(data.neural_trials, data.behavior_trials):
        X_trial = create_lagged_features(neural, lags)
        if max_lag_future == 0:
            X_valid = X_trial
            Y_valid = behavior
        else:
            X_valid = X_trial[:-max_lag_future]
            Y_valid = behavior[:-max_lag_future]
        X_parts.append(X_valid)
        Y_parts.append(Y_valid)
    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    if fit_scaler:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all)
    elif scaler is not None:
        X_all = scaler.transform(X_all)
    return X_all, Y_all, scaler


class TrialDataset(Dataset):
    def __init__(self, data: TrialData, x_scaler):
        self.neural_trials = [torch.tensor(x_scaler.transform(n), dtype=torch.float32) for n in data.neural_trials]
        self.behavior_trials = [torch.tensor(b, dtype=torch.float32) for b in data.behavior_trials]
    def __len__(self):
        return len(self.neural_trials)
    def __getitem__(self, idx):
        return self.neural_trials[idx], self.behavior_trials[idx]

def collate_fn(batch):
    neurals, behaviors = zip(*batch)
    lengths = torch.tensor([len(n) for n in neurals], dtype=torch.int64)
    neurals_padded = pad_sequence(neurals, batch_first=True)
    behaviors_padded = pad_sequence(behaviors, batch_first=True)
    return neurals_padded, behaviors_padded, lengths

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNDecoderSimple(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            padding = (kernel_size - 1) * dilation
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, 1)
        
    def forward(self, x, lengths=None):
        out = x.transpose(1, 2)
        for layer in self.layers:
            out = layer(out)
        out = self.output_layer(out)
        return out.transpose(1, 2)

class TCNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.relu(self.chomp1(self.conv1(x))))
        out = self.dropout(self.relu(self.chomp2(self.conv2(out))))
        return self.relu(out + self.downsample(x))

class TCNDecoderResidual(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            self.blocks.append(TCNResidualBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, 1)
        
    def forward(self, x, lengths=None):
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        out = self.output_layer(out)
        return out.transpose(1, 2)


def run_glm(train_data, valid_data, test_data):
    LAGS = list(range(0, 5))  # causal
    label = "GLM"
    ALPHA_GRID = np.logspace(-3, 2, 12)
    
    print(f"[{label}] Preparing data (Lags: {LAGS})...")
    X_train, Y_train, scaler = prepare_design_matrix(train_data, LAGS, fit_scaler=True)
    X_valid, Y_valid, _ = prepare_design_matrix(valid_data, LAGS, scaler=scaler)
    
    best_alpha = None
    best_val_mse = np.inf

    Y_mean = Y_train.mean(axis=0)
    Y_train_centered = Y_train - Y_mean
    XtX = X_train.T @ X_train          # (D, D)
    XtY = X_train.T @ Y_train_centered  # (D, 3)
    eigvals, eigvecs = np.linalg.eigh(XtX)  # O(D^3)

    def ridge_predict_eigh(X, alpha, intercept):
        scale = 1.0 / (eigvals + alpha)          # (D,)
        w = eigvecs @ (scale[:, None] * (eigvecs.T @ XtY))  # (D, 3)
        return X @ w + intercept

    for alpha in ALPHA_GRID:
        Y_val_pred = ridge_predict_eigh(X_valid, alpha, Y_mean)
        val_mse = mean_squared_error(Y_valid, Y_val_pred)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha

    print(f"[{label}] Best Alpha: {best_alpha:.4f}, Val MSE: {best_val_mse:.4f}")

    scale = 1.0 / (eigvals + best_alpha)
    w_final = eigvecs @ (scale[:, None] * (eigvecs.T @ XtY))  # (D, 3)
    intercept_final = Y_mean
    
    Y_pred_trials = []
    Y_true_trials = []
    max_lag_future = max(0, -min(LAGS)) if LAGS else 0
    
    for neural, behavior in zip(test_data.neural_trials, test_data.behavior_trials):
        X_trial = create_lagged_features(neural, LAGS)
        if max_lag_future == 0:
            X_curr = X_trial
            Y_curr = behavior
        else:
            X_curr = X_trial[:-max_lag_future]
            Y_curr = behavior[:-max_lag_future]
        X_curr = scaler.transform(X_curr)
        Y_p = X_curr @ w_final + intercept_final
        Y_pred_trials.append(Y_p)
        Y_true_trials.append(Y_curr)
        
    return evaluate_predictions(Y_true_trials, Y_pred_trials, label)

def run_tcn(train_data, valid_data, test_data, x_scaler, device, model_cls, hidden_dim, label):
    TCN_KERNEL_SIZE = 5
    TCN_LAYERS = 4
    TCN_DROPOUT = 0.2
    TCN_EPOCHS = 200
    TCN_BATCH_SIZE = 32
    TCN_LRS = [1e-3]
    TCN_WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3]
    PATIENCE = 20
    
    print(f"[{label}] Preparing data...")
    train_dataset = TrialDataset(train_data, x_scaler)
    valid_dataset = TrialDataset(valid_data, x_scaler)
    train_loader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=TCN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    input_dim = train_data.neural_dim
    output_dim = train_data.behavior_dim
    
    best_val_loss = float("inf")
    best_model_state = None
    best_config = None
    
    config_idx = 0
    total_configs = len(TCN_LRS) * len(TCN_WEIGHT_DECAYS)
    for lr in TCN_LRS:
        for wd in TCN_WEIGHT_DECAYS:
            config_idx += 1
            print(f"[{label}] Config {config_idx}/{total_configs}: LR={lr}, WD={wd}")
            model = model_cls(input_dim, hidden_dim, output_dim, TCN_LAYERS, TCN_KERNEL_SIZE, TCN_DROPOUT).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.MSELoss(reduction='none')
            
            run_best_val_loss = float("inf")
            run_best_state = None
            patience_counter = 0
            
            for epoch in range(TCN_EPOCHS):
                # Train
                model.train()
                for bx, by, lengths in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    max_len = bx.size(1)
                    mask = torch.arange(max_len, device=device)[None, :] < lengths.to(device)[:, None]
                    mask = mask.unsqueeze(-1)
                    loss = (criterion(pred, by) * mask).sum() / mask.sum()
                    loss.backward()
                    optimizer.step()
                
                # Validate
                model.eval()
                val_loss = 0.0
                total_val = 0
                with torch.no_grad():
                    for bx, by, lengths in valid_loader:
                        bx, by = bx.to(device), by.to(device)
                        pred = model(bx)
                        max_len = bx.size(1)
                        mask = torch.arange(max_len, device=device)[None, :] < lengths.to(device)[:, None]
                        mask = mask.unsqueeze(-1)
                        val_loss += ((criterion(pred, by) * mask).sum().item())
                        total_val += mask.sum().item()
                val_loss /= total_val
                
                # Per-run early stopping
                if val_loss < run_best_val_loss:
                    run_best_val_loss = val_loss
                    run_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"[{label}]   Early stop at epoch {epoch+1}, best val_loss={run_best_val_loss:.6f}")
                        break
            
            if run_best_val_loss < best_val_loss:
                best_val_loss = run_best_val_loss
                best_model_state = run_best_state
                best_config = {"lr": lr, "wd": wd}
            
    print(f"[{label}] Best Config: LR={best_config['lr']}, WD={best_config['wd']}, Val Loss: {best_val_loss:.4f}")
    
    final_model = model_cls(input_dim, hidden_dim, output_dim, TCN_LAYERS, TCN_KERNEL_SIZE, TCN_DROPOUT).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()
    
    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"[{label}] Model parameters: {n_params:,}")
    
    Y_pred_trials = []
    Y_true_trials = []
    
    with torch.no_grad():
        for neural, behavior in zip(test_data.neural_trials, test_data.behavior_trials):
            neural_scaled = x_scaler.transform(neural)
            inputs_tensor = torch.from_numpy(neural_scaled).float().unsqueeze(0).to(device)
            pred = final_model(inputs_tensor).cpu().numpy()[0]
            Y_pred_trials.append(pred)
            Y_true_trials.append(behavior)
            
    return evaluate_predictions(Y_true_trials, Y_pred_trials, label)


if __name__ == "__main__":
    if len(SESSION_PATHS) == 0:
        print("WARNING: SESSION_PATHS is empty. Please edit the script to add .npz files.")
        exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_KEYS = ["GLM", "TCN_Simple64", "TCN_Res64", "TCN_Res128"]
    global_stats = {
        k: {"avg_corr": [], "avg_r2": [], "avg_mse": [], "dim_corr": [], "dim_r2": [], "dim_mse": []}
        for k in MODEL_KEYS
    }

    for i, session_path in enumerate(SESSION_PATHS):
        path_obj = Path(session_path)
        print("="*60)
        print(f"Processing Session {i+1}/{len(SESSION_PATHS)}: {path_obj.name}")
        print("="*60)
        
        try:
            train_data, valid_data, test_data = load_session_data(path_obj)
        except Exception as e:
            print(f"Error loading {path_obj}: {e}")
            continue

        X_train_raw = np.concatenate(train_data.neural_trials, axis=0)
        x_scaler = StandardScaler().fit(X_train_raw)
            
        # 1. GLM
        glm_metrics = run_glm(train_data, valid_data, test_data)
        for k in global_stats["GLM"].keys():
            global_stats["GLM"][k].append(glm_metrics[k])

        # 2. TCN Simple (hidden=64, no residual)
        m = run_tcn(train_data, valid_data, test_data, x_scaler, device,
                    model_cls=TCNDecoderSimple, hidden_dim=64, label="TCN_Simple64")
        for k in global_stats["TCN_Simple64"].keys():
            global_stats["TCN_Simple64"][k].append(m[k])

        # 3. TCN Residual (hidden=64)
        m = run_tcn(train_data, valid_data, test_data, x_scaler, device,
                    model_cls=TCNDecoderResidual, hidden_dim=64, label="TCN_Res64")
        for k in global_stats["TCN_Res64"].keys():
            global_stats["TCN_Res64"][k].append(m[k])

        # 4. TCN Residual (hidden=128)
        m = run_tcn(train_data, valid_data, test_data, x_scaler, device,
                    model_cls=TCNDecoderResidual, hidden_dim=128, label="TCN_Res128")
        for k in global_stats["TCN_Res128"].keys():
            global_stats["TCN_Res128"][k].append(m[k])
            
    print("\n" + "="*60)
    print("FINAL GLOBAL STATISTICS (Mean ± Std over sessions)")
    print("="*60)
    
    for model_name, metrics in global_stats.items():
        if len(metrics["avg_corr"]) == 0:
            print(f"{model_name}: No results.")
            continue
            
        print(f"--- {model_name} ---")
        print("  [Overall]")
        for k in ["avg_corr", "avg_r2", "avg_mse"]:
            vals = np.array(metrics[k])
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1)
            print(f"    {k}: {mean_val:.4f} ± {std_val:.4f}")
            
        print("  [Per-Dimension]")
        for k_dim in ["dim_corr", "dim_r2", "dim_mse"]:
            list_of_vecs = metrics[k_dim]
            shapes = [v.shape for v in list_of_vecs]
            if not all(s == shapes[0] for s in shapes):
                print(f"    {k_dim}: Cannot aggregate, dimension mismatch across sessions {shapes}")
                continue
            stacked = np.stack(list_of_vecs, axis=0)
            mean_vec = np.mean(stacked, axis=0)
            std_vec = np.std(stacked, axis=0, ddof=1)
            mean_str = np.array2string(mean_vec, precision=4, floatmode='fixed', suppress_small=True)
            std_str = np.array2string(std_vec, precision=4, floatmode='fixed', suppress_small=True)
            print(f"    {k_dim} Mean: {mean_str}")
            print(f"    {k_dim} Std:  {std_str}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE: R² (mean ± std across sessions)")
    print("=" * 80)

    n_dims = 0
    for mk in MODEL_KEYS:
        if len(global_stats[mk]["dim_r2"]) > 0:
            n_dims = global_stats[mk]["dim_r2"][0].shape[0]
            break

    dim_headers = [f"Dim{d}" for d in range(n_dims)]
    header = f"{'Method':<18}" + "".join(f"{dh:>20}" for dh in dim_headers) + f"{'Macro':>20}"
    print(header)
    print("-" * (18 + 20 * (n_dims + 1)))

    for mk in MODEL_KEYS:
        dim_r2_list = global_stats[mk]["dim_r2"]
        if len(dim_r2_list) == 0:
            print(f"{mk:<18}" + "  N/A" * (n_dims + 1))
            continue
        arr = np.stack(dim_r2_list, axis=0)  # (n_sessions, n_dims)
        macro = np.mean(arr, axis=1)          # (n_sessions,)
        row = f"{mk:<18}"
        for d in range(n_dims):
            m, s = np.mean(arr[:, d]), np.std(arr[:, d], ddof=1)
            row += f"{m:>10.4f}±{s:<8.4f}"
        mm, ms = np.mean(macro), np.std(macro, ddof=1)
        row += f"{mm:>10.4f}±{ms:<8.4f}"
        print(row)

    print("=" * 80)
