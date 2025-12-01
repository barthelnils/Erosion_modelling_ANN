import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from sklearn.preprocessing import StandardScaler

def expand_fields_individually(d, expand_pixels):
    mask = d["mask"].astype(bool)
    labeled, n_fields = label(mask)
    expanded_mask = np.zeros_like(mask, bool)

    for i in range(1, n_fields + 1):
        expanded_mask |= binary_dilation(labeled == i, iterations=expand_pixels)

    new_zone = expanded_mask & ~mask

    d_exp = d.copy()
    d_exp["data"]   = d["data"].copy()
    d_exp["target"] = d["target"].copy()
    d_exp["data"][new_zone]   = 0
    d_exp["target"][new_zone] = 0
    d_exp["mask"] = expanded_mask
    return d_exp


def fit_scaler_for_cnn(train_areas):
    mats = []
    for d in train_areas:
        X = d["data"][d["mask"]].reshape(-1, d["data"].shape[-1])
        nonzero = np.any(X != 0, axis=1)
        mats.append(X[nonzero])
    X_all = np.vstack(mats)
    return StandardScaler().fit(X_all)


def apply_scaler(d, scaler):
    H, W, C = d["data"].shape
    flat = d["data"].reshape(-1, C)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(H, W, C)


def compute_safe_center_mask(mask, patch_size):
    if not np.any(mask):
        return mask
    radius = min(patch_size // 2, 2)
    size = 2*radius + 1
    st = np.ones((size, size), bool)
    return binary_erosion(mask, structure=st)


def extract_patches(data_std, target, center_mask, patch_size):
    pad = patch_size // 2
    data_pad = np.pad(data_std, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    target_pad = np.pad(target, ((pad, pad), (pad, pad)), mode="reflect")

    idx = np.argwhere(center_mask)
    N = idx.shape[0]
    
    C = data_std.shape[-1]
    X = np.zeros((N, patch_size, patch_size, C), np.float32)
    y = np.zeros(N, np.float32)

    for k, (r, c) in enumerate(idx):
        X[k] = data_pad[r:r+patch_size, c:c+patch_size, :]
        y[k] = target_pad[r+pad, c+pad]

    return X, y, idx
