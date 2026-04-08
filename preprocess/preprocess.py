import os
import numpy as np
from pynwb import NWBHDF5IO
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter, lfilter_zi


filepath = None # required for the datapath
eps = 1e-6
RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
TEST_SHARE_WITH_VAL = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
print('TEST_SHARE_WITH_VAL: {}'.format(TEST_SHARE_WITH_VAL))
EMA_ALPHA = 0.15
MIN_TRIAL_LEN = 60
MAX_TRIAL_LEN = 200
dropped_num = 0

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pad_2d(arr, target_len):
    T, F = arr.shape
    if T < target_len: return np.vstack([arr, np.zeros((target_len - T, F), dtype=arr.dtype)])
    return arr[:target_len, :]

def split_trials(trial_ids, random_state=42):
    all_idx = np.arange(len(trial_ids))
    train_idx, temp_idx = train_test_split(all_idx, test_size=(1.0 - TRAIN_RATIO), random_state=random_state, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=TEST_SHARE_WITH_VAL, random_state=random_state, shuffle=True)
    return train_idx, val_idx, test_idx

def compute_train_zscore_stats(trials_list, train_idx):
    X_train = np.concatenate([trials_list[i] for i in train_idx], axis=0)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + eps
    return mean, std

def zscore_trials(trials_list, mean, std):
    return [(x - mean) / std for x in trials_list]

def build_ar_split(trials_X, trials_Y, lengths, trial_ids, idx_set, max_len):
    sel_X = [trials_X[i] for i in idx_set]
    sel_Y = [trials_Y[i] for i in idx_set]
    sel_L = lengths[idx_set]
    sel_trial_ids = trial_ids[idx_set]
    padded_X = np.array([pad_2d(x, max_len).T for x in sel_X], dtype=np.float32)
    padded_Y = np.array([pad_2d(y, max_len).T for y in sel_Y], dtype=np.float32)
    return dict(padded_X=padded_X, padded_Y=padded_Y, lengths=sel_L, trial_ids=sel_trial_ids)

def parse_plane_id(p_name: str, fallback: int) -> int:
    try: return int(str(p_name).split("_")[-1])
    except Exception: return fallback

def ema_causal_2d(x: np.ndarray, alpha: float = 0.15, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    b = np.array([alpha], dtype=np.float32)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)
    zi0 = lfilter_zi(b, a).astype(np.float32)
    x0 = np.take(x, indices=0, axis=axis)
    zi = zi0.reshape((1,) + (1,) * x0.ndim) * x0[np.newaxis, ...]
    y, _ = lfilter(b, a, x, axis=axis, zi=zi)
    return y.astype(np.float32)


io = NWBHDF5IO(filepath, 'r')
nwb = io.read()
imgseg = nwb.processing['ophys']['ImageSegmentation']
ps_names = list(imgseg.plane_segmentations.keys())
base = os.path.basename(filepath)
stem = os.path.splitext(base)[0]
out_dir = (os.path.dirname(filepath) or ".") + "_processed"
ensure_dir(out_dir)
frame_aligned_velocity = nwb.processing['behavior']['frame_aligned_velocity']['frame_aligned_pitch_roll_yaw_velocity'].data[:].astype(np.float32) 
frame_aligned_trial_number = nwb.processing['behavior']['frame_aligned_trial_number'].data[:]
plane_idx_for_imaging_frames = nwb.processing['behavior']['plane_idx_for_imaging_frames'].data[:]
trial_all = np.asarray(frame_aligned_trial_number)
valid_trial_ids = np.unique(trial_all[np.isfinite(trial_all)]).astype(int)
valid_trial_ids = np.sort(valid_trial_ids)
print("Valid trials (global):", len(valid_trial_ids))
plane_infos = []
roi_meta_parts = []

for k, p_name in enumerate(ps_names):
    plane_id = parse_plane_id(p_name, fallback=k)
    planeSeg_df = imgseg.plane_segmentations[p_name].to_dataframe()
    roi_local_id = planeSeg_df.index.to_numpy()
    roi_area = planeSeg_df["area"].astype(str).to_numpy()
    roi_mTagBFP2 = planeSeg_df["mTagBFP2"].astype(np.uint8).to_numpy()
    roi_mScarlet = planeSeg_df["mScarlet"].astype(np.uint8).to_numpy()
    df_series = nwb.processing['ophys'][f'df_over_f_plane_{plane_id}'][f'dF_over_F_plane_{plane_id}']
    df_data = df_series.data[:].astype(np.float32)
    idx_global = np.where(np.asarray(plane_idx_for_imaging_frames) == plane_id)[0]
    if df_data.shape[0] != len(idx_global): raise RuntimeError(f"Plane {plane_id} mismatch: df={df_data.shape[0]} vs idx_global={len(idx_global)}")
    plane_infos.append(dict(
        plane_id=plane_id,
        df_data=df_data, 
        idx_global=idx_global,
        roi_local_id=roi_local_id,
        roi_area=roi_area,
        roi_mTagBFP2=roi_mTagBFP2,
        roi_mScarlet=roi_mScarlet,
    ))
    roi_meta_parts.append((plane_id, df_data.shape[1], roi_local_id, roi_area, roi_mTagBFP2, roi_mScarlet))

plane_infos.sort(key=lambda d: d["plane_id"])
N_each = [d["df_data"].shape[1] for d in plane_infos]
N_total = int(sum(N_each))
roi_plane = np.concatenate([np.full(n, d["plane_id"], dtype=np.int32) for d, n in zip(plane_infos, N_each)], axis=0)
roi_within_plane = np.concatenate([np.arange(n, dtype=np.int32) for n in N_each], axis=0)
roi_local_id_all = np.concatenate([d["roi_local_id"] for d in plane_infos], axis=0)
roi_area_all = np.concatenate([d["roi_area"] for d in plane_infos], axis=0)
roi_mTagBFP2_all = np.concatenate([d["roi_mTagBFP2"] for d in plane_infos], axis=0)
roi_mScarlet_all = np.concatenate([d["roi_mScarlet"] for d in plane_infos], axis=0)
meta_path = os.path.join(out_dir, f"{stem}_roi_meta_allplanes.npz")
np.savez(
    meta_path,
    roi_global_id=np.arange(N_total, dtype=np.int64),
    roi_plane=roi_plane,
    roi_within_plane=roi_within_plane,
    roi_local_id=roi_local_id_all,
    area=roi_area_all.astype(str),
    mTagBFP2=roi_mTagBFP2_all.astype(np.uint8),
    mScarlet=roi_mScarlet_all.astype(np.uint8),
)
print("Saved combined ROI meta:", meta_path, "| N_total =", N_total)

trials_X_raw = []
trials_Y_raw = []
lengths = []
kept_trial_ids = []
for tid in valid_trial_ids:
    X_parts = []
    F_parts = []
    ok = True
    for d in plane_infos:
        idxg = d["idx_global"]
        tnums_p = trial_all[idxg]
        mask = (tnums_p == tid)
        Xp = d["df_data"][mask, :]
        fp = idxg[mask]
        if Xp.shape[0] < MIN_TRIAL_LEN or Xp.shape[0] > MAX_TRIAL_LEN:
            ok = False
            print(Xp.shape[0])
            dropped_num += 1
            break
        X_parts.append(Xp)
        F_parts.append(fp)
    if not ok: continue
    L = min(x.shape[0] for x in X_parts)
    X_trial = np.concatenate([x[:L] for x in X_parts], axis=1)
    Y_trial = np.zeros((L, 3), dtype=np.float32)
    for t in range(L):
        frame_ids = np.array([fp[t] for fp in F_parts], dtype=np.int64)
        Y_trial[t] = frame_aligned_velocity[frame_ids].mean(axis=0)
    trials_X_raw.append(X_trial.astype(np.float32))
    trials_Y_raw.append(Y_trial.astype(np.float32))
    lengths.append(L)
    kept_trial_ids.append(int(tid))

if len(trials_X_raw) == 0: raise RuntimeError("No usable trials after combining planes (check trial numbers and masks).")
lengths = np.asarray(lengths, dtype=np.int64)
kept_trial_ids = np.asarray(kept_trial_ids, dtype=np.int64)
print("Built combined trials:", len(trials_X_raw), "| length min/mean/max =", lengths.min(), float(lengths.mean()), lengths.max())
train_idx, val_idx, test_idx = split_trials(kept_trial_ids, random_state=RANDOM_STATE)
print(f"Split trials: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
trials_X_smooth = [ema_causal_2d(x, alpha=EMA_ALPHA, axis=0) for x in trials_X_raw]
cal_mean, cal_std = compute_train_zscore_stats(trials_X_smooth, train_idx)
trials_X_z = zscore_trials(trials_X_smooth, cal_mean, cal_std)
beh_mean, beh_std = compute_train_zscore_stats(trials_Y_raw, train_idx) 
trials_Y_z = zscore_trials(trials_Y_raw, beh_mean, beh_std)
max_len = int(lengths.max())
train_ar = build_ar_split(trials_X_z, trials_Y_z, lengths, kept_trial_ids, train_idx, max_len)
val_ar   = build_ar_split(trials_X_z, trials_Y_z, lengths, kept_trial_ids, val_idx,   max_len)
test_ar  = build_ar_split(trials_X_z, trials_Y_z, lengths, kept_trial_ids, test_idx,  max_len)
ar_path = os.path.join(out_dir, f"{stem}_AR_70_15_15_allplanes.npz")
np.savez(
    ar_path,
    train_padded_X=train_ar["padded_X"],
    train_padded_Y=train_ar["padded_Y"],
    train_lengths=train_ar["lengths"],
    train_trial_ids=train_ar["trial_ids"],
    val_padded_X=val_ar["padded_X"],
    val_padded_Y=val_ar["padded_Y"],
    val_lengths=val_ar["lengths"],
    val_trial_ids=val_ar["trial_ids"],
    test_padded_X=test_ar["padded_X"],
    test_padded_Y=test_ar["padded_Y"],
    test_lengths=test_ar["lengths"],
    test_trial_ids=test_ar["trial_ids"],
    z_mean=cal_mean.astype(np.float32),
    z_std=cal_std.astype(np.float32),
    beh_z_mean=beh_mean.astype(np.float32),
    beh_z_std=beh_std.astype(np.float32),
    max_len=np.int64(max_len),
    ema_alpha=np.float32(EMA_ALPHA),
    n_total=np.int64(N_total),
)
print("Saved AR (combined):", ar_path, "| train X:", train_ar["padded_X"].shape, "| train Y:", train_ar["padded_Y"].shape)
io.close()
print("\nDone.")
print("Dropped (some plane < MIN_TRIAL_LEN):", dropped_num)
