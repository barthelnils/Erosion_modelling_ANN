"""
LOAO grid search pipeline for soil erosion modelling.
ally.

Usage example (local):
    python loao_grid_search.py --config config.yaml

"""

import os
import math
import json
import argparse
import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import rasterio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestRegressor

from scipy.ndimage import binary_erosion, binary_dilation, label

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

import yaml

warnings.filterwarnings("ignore", category=UserWarning)

# Config helpers


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_param_grid(grid_dict):
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def sample_param_grid(grid_dict, n_random, seed):
    all_combos = list(generate_param_grid(grid_dict))
    if n_random is None or n_random <= 0 or n_random >= len(all_combos):
        return all_combos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_combos), size=int(n_random), replace=False)
    return [all_combos[i] for i in idx]

def safe_float(x):
    # YAML null -> None is fine; strings like "1e-4" should be floatable
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x))

def cfg_key(model_type, params):
    # stable key for selection counts
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        if v is None:
            parts.append(f"{k}None")
        else:
            parts.append(f"{k}{v}")
    return f"{model_type}_" + "_".join(parts)



# Data I/O

def read_study_area_data(data_dir, area, band_names):
    path = os.path.join(data_dir, area + ".tif")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing raster: " + path)

    with rasterio.open(path) as src:
        target = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

        desc = src.descriptions
        predictors = []

        if desc is not None and any([d is not None for d in desc]):
            for i, d in enumerate(desc, start=1):
                if i == 1:
                    continue
                if d in band_names:
                    predictors.append(src.read(i).astype(np.float32))
        else:
            for i in range(2, src.count + 1):
                predictors.append(src.read(i).astype(np.float32))

        if len(predictors) == 0:
            raise RuntimeError("No predictor bands found in " + path)

        data = np.stack(predictors, axis=-1)
        feat_valid = np.all(np.isfinite(data), axis=-1)
        mask = (target != nodata) & (target >= 0) & feat_valid

        profile = src.profile

    return {
        "data": data,
        "target": target,
        "mask": mask,
        "shape": target.shape,
        "profile": profile
    }

def loao_splits(areas):
    for a in areas:
        yield a, [x for x in areas if x != a]


# Metrics

def categorize_soil_loss(v):
    if v == 0:
        return 0
    if v < 0.25:
        return 1
    if v < 1.0:
        return 2
    if v < 2.0:
        return 3
    if v < 5.0:
        return 4
    return 5

def classification_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"f1": np.nan, "precision": np.nan, "recall": np.nan, "accuracy": np.nan}
    cat_t = np.vectorize(categorize_soil_loss)(y_true)
    cat_p = np.vectorize(categorize_soil_loss)(y_pred)
    return {
        "f1": float(f1_score(cat_t, cat_p, average="weighted", zero_division=0)),
        "precision": float(precision_score(cat_t, cat_p, average="weighted", zero_division=0)),
        "recall": float(recall_score(cat_t, cat_p, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(cat_t, cat_p))
    }

def regression_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

def spatial_correlation_full(y_true_full, y_pred_full, mask):
    valid = mask & np.isfinite(y_true_full) & np.isfinite(y_pred_full)
    if int(np.sum(valid)) < 2:
        return {"spatial_corr": np.nan}
    corr = np.corrcoef(y_true_full[valid], y_pred_full[valid])[0, 1]
    return {"spatial_corr": float(corr)}

def weighted_rmse(rmses, ns):
    rmses = np.asarray(rmses, dtype=float)
    ns = np.asarray(ns, dtype=float)
    if ns.sum() <= 0:
        return float("nan")
    w = ns / ns.sum()
    return float(np.sqrt(np.sum(w * (rmses ** 2))))

def weighted_mean(vals, ns):
    vals = np.asarray(vals, dtype=float)
    ns = np.asarray(ns, dtype=float)
    if ns.sum() <= 0:
        return float("nan")
    w = ns / ns.sum()
    return float(np.sum(w * vals))


# RF / SNN / DNN per-pixel core


def fit_scaler_pixel(train_areas, info):
    mats = []
    for a in train_areas:
        d = info[a]
        X = d["data"][d["mask"]]
        mats.append(X)
    Xall = np.vstack(mats)
    return StandardScaler().fit(Xall)

def build_rf(params, seed):
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] in [None, "null", "None"] else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=None if params["max_features"] in [None, "null", "None"] else params["max_features"],
        n_jobs=-1,
        random_state=seed
    )

def build_snn(input_dim, params):
    lr = safe_float(params["lr"])
    l2_dense = safe_float(params["l2_dense"])
    units = int(params["units"])
    L2r = regularizers.l2(l2_dense)
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units, activation="relu", kernel_regularizer=L2r),
        layers.Dense(1, activation="relu")
    ])
    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m

def build_dnn(input_dim, params):
    lr = safe_float(params["lr"])
    l2_dense = safe_float(params["l2_dense"])
    layers_n = int(params["layers"])
    units = int(params["units"])
    dropout = float(params["dropout"])
    L2r = regularizers.l2(l2_dense)
    m = models.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(layers_n):
        m.add(layers.Dense(units, activation="relu", kernel_regularizer=L2r))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="relu"))
    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m

def train_eval_pixel_model(model_type, params, train_areas, test_area, info, nn_train_cfg, seed, do_early_stop): # Inner areas for training only
    scaler = fit_scaler_pixel(train_areas, info)

    # train data
    Xtr = np.vstack([scaler.transform(info[a]["data"][info[a]["mask"]]) for a in train_areas])
    ytr = np.concatenate([info[a]["target"][info[a]["mask"]] for a in train_areas])

    # test data
    m_te = info[test_area]["mask"]
    Xte = scaler.transform(info[test_area]["data"][m_te])
    yte = info[test_area]["target"][m_te]

    ytr_log = np.log1p(ytr)

    best_epoch = None

    if model_type == "rf":
        model = build_rf(params, seed)
        model.fit(Xtr, ytr_log)
        yp = np.expm1(model.predict(Xte))
        yp = np.nan_to_num(yp)
        rmse = math.sqrt(mean_squared_error(yte, yp))
        mae = mean_absolute_error(yte, yp)
        r2 = r2_score(yte, yp)
        return float(rmse), float(mae), float(r2), int(yte.size), best_epoch

    # NN
    batch_size = int(nn_train_cfg.get("batch_size", 2048))
    epochs = int(nn_train_cfg.get("epochs", 50))
    patience = int(nn_train_cfg.get("patience", 5))
    shuffle_buf = int(nn_train_cfg.get("shuffle_buf", 10000))

    if model_type == "snn":
        model = build_snn(Xtr.shape[1], params)
    elif model_type == "dnn":
        model = build_dnn(Xtr.shape[1], params)
    else:
        raise ValueError("Unknown pixel model type: " + model_type)

    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr_log)) \
        .shuffle(min(Xtr.shape[0], shuffle_buf), seed=seed, reshuffle_each_iteration=True) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_te = tf.data.Dataset.from_tensor_slices((Xte, np.log1p(yte))) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if do_early_stop:
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        hist = model.fit(ds_tr, validation_data=ds_te, epochs=epochs, callbacks=[es], verbose=0)
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
    else:
        fixed_epochs = int(nn_train_cfg["fixed_epochs"])
        model.fit(ds_tr, epochs=fixed_epochs, verbose=0)

    yp = np.expm1(model.predict(ds_te, verbose=0).flatten())
    yp = np.nan_to_num(yp)

    rmse = math.sqrt(mean_squared_error(yte, yp))
    mae = mean_absolute_error(yte, yp)
    r2 = r2_score(yte, yp)

    tf.keras.backend.clear_session()
    return float(rmse), float(mae), float(r2), int(yte.size), best_epoch


# CNN patch logic (based on your old workflow style)
def expand_fields_individually(d, expand_pixels):
    mask = d["mask"].astype(bool)
    labeled, n_fields = label(mask)
    expanded_mask = np.zeros_like(mask, dtype=bool)
    for i in range(1, n_fields + 1):
        expanded_mask |= binary_dilation(labeled == i, iterations=int(expand_pixels))
    new_zone = expanded_mask & ~mask

    data_exp = np.copy(d["data"])
    target_exp = np.copy(d["target"])
    data_exp[new_zone] = 0.0
    target_exp[new_zone] = 0.0

    out = d.copy()
    out["data"] = data_exp
    out["target"] = target_exp
    out["mask"] = expanded_mask
    out["shape"] = target_exp.shape
    return out

def fit_scaler_for_cnn(expanded_train_list):
    arrs = []
    for d in expanded_train_list:
        if not np.any(d["mask"]):
            continue
        X = d["data"][d["mask"]].reshape(-1, d["data"].shape[-1])
        nonzero = np.any(X != 0.0, axis=1)
        X = X[nonzero]
        if X.size > 0:
            arrs.append(X)
    if len(arrs) == 0:
        raise RuntimeError("No valid pixels to fit CNN scaler.")
    X_all = np.vstack(arrs)
    return StandardScaler().fit(X_all)

def apply_scaler_to_data(d, scaler):
    H, W, C = d["data"].shape
    flat = d["data"].reshape(-1, C)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(H, W, C)

def compute_safe_center_mask(mask, patch_size):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    # require full patch to fit -> erode with full patch footprint
    structure = np.ones((int(patch_size), int(patch_size)), dtype=bool)
    return binary_erosion(mask, structure=structure, border_value=0)

def extract_patches(data_std, target, center_mask, patch_size):
    pad = int(patch_size) // 2
    data_std_pad = np.pad(data_std, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    target_pad = np.pad(target, ((pad, pad), (pad, pad)), mode="reflect")

    idx = np.argwhere(center_mask)
    N = idx.shape[0]
    C = data_std.shape[-1]
    X = np.zeros((N, int(patch_size), int(patch_size), C), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)

    for k, (r, c) in enumerate(idx):
        X[k] = data_std_pad[r:r + int(patch_size), c:c + int(patch_size), :]
        y[k] = target_pad[r + pad, c + pad]

    return X, y, idx

def build_cnn(input_shape, params):
    lr = safe_float(params["lr"])
    l2_conv = safe_float(params["l2_conv"])
    l2_dense = safe_float(params["l2_dense"])
    conv_layers = int(params["conv_layers"])
    filters = int(params["filters"])
    dropout = float(params["dropout"])
    kernel = int(params["kernel_size"])
    dense_units = int(params.get("dense_units", 256))  # optional

    L2c = regularizers.l2(l2_conv)
    L2d = regularizers.l2(l2_dense)

    inp = layers.Input(shape=input_shape)
    x = inp

    # conv blocks
    for i in range(conv_layers):
        x = layers.Conv2D(filters, (kernel, kernel), activation="relu",
                          padding="same", kernel_regularizer=L2c)(x)
        if (i + 1) % 2 == 0:
            x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=L2d)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="relu")(x)

    m = models.Model(inputs=inp, outputs=out)
    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m

def train_eval_cnn(params, train_areas, test_area, info, nn_train_cfg, seed, do_early_stop):
    patch_size = int(params["patch_size"])
    expand_pixels = int(params.get("expand_pixels", patch_size // 2 + 1))

    # expand training areas
    expanded_train = [expand_fields_individually(info[a], expand_pixels) for a in train_areas]
    scaler = fit_scaler_for_cnn(expanded_train)

    Xtr_list, ytr_list = [], []
    for d in expanded_train:
        if not np.any(d["mask"]):
            continue
        data_std = apply_scaler_to_data(d, scaler)
        safe = compute_safe_center_mask(d["mask"], patch_size)
        Xp, yp, _ = extract_patches(data_std, d["target"], safe, patch_size)
        if Xp.shape[0] == 0:
            continue
        nonzero = ~np.all(Xp == 0.0, axis=(1, 2, 3))
        Xp = Xp[nonzero]
        yp = yp[nonzero]
        if Xp.shape[0] > 0:
            Xtr_list.append(Xp)
            ytr_list.append(yp)

    if len(Xtr_list) == 0:
        return float("nan"), float("nan"), float("nan"), 0, None

    Xtr = np.concatenate(Xtr_list, axis=0)
    ytr = np.concatenate(ytr_list, axis=0)

    # expanded validation/test
    val_exp = expand_fields_individually(info[test_area], expand_pixels)
    if not np.any(val_exp["mask"]):
        return float("nan"), float("nan"), float("nan"), 0, None

    data_std_v = apply_scaler_to_data(val_exp, scaler)
    safe_v = compute_safe_center_mask(val_exp["mask"], patch_size)
    Xv, yv, idx = extract_patches(data_std_v, val_exp["target"], safe_v, patch_size)

    if Xv.shape[0] == 0:
        return float("nan"), float("nan"), float("nan"), 0, None

    # train
    batch_size = int(nn_train_cfg.get("batch_size", 256))
    epochs = int(nn_train_cfg.get("epochs", 80))
    patience = int(nn_train_cfg.get("patience", 8))
    shuffle_buf = int(nn_train_cfg.get("shuffle_buf", 10000))

    ytr_log = np.log1p(ytr)
    yv_log = np.log1p(yv)

    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr_log)) \
        .shuffle(min(Xtr.shape[0], shuffle_buf), seed=seed, reshuffle_each_iteration=True) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices((Xv, yv_log)) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    input_shape = (patch_size, patch_size, Xtr.shape[-1])
    model = build_cnn(input_shape, params)

    best_epoch = None
    if do_early_stop:
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=[es], verbose=0)
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
    else:
        fixed_epochs = int(nn_train_cfg["fixed_epochs"])
        model.fit(ds_tr, epochs=fixed_epochs, verbose=0)

    yp = np.expm1(model.predict(ds_va, verbose=0).reshape(-1))
    yp = np.nan_to_num(yp)

    rmse = math.sqrt(mean_squared_error(yv, yp))
    mae = mean_absolute_error(yv, yp)
    r2 = r2_score(yv, yp)

    tf.keras.backend.clear_session()
    return float(rmse), float(mae), float(r2), int(len(yv)), best_epoch


# Nested LOAO

def inner_score(model_type, params, outer_train_areas, info, nn_train_cfg, seed):
    """
    Inner LOAO on outer_train_areas.
    Returns:
      inner_w_rmse, inner_w_mae, median_epoch (epoch only for NN/CNN else None)
    """
    rmses, maes, ns, epochs = [], [], [], []

    for inner_val in outer_train_areas:
        inner_train = [a for a in outer_train_areas if a != inner_val]

        if model_type in ["rf", "snn", "dnn"]:
            rmse, mae, _r2, n, best_ep = train_eval_pixel_model(
                model_type, params, inner_train, inner_val, info,
                nn_train_cfg, seed, do_early_stop=(model_type != "rf")
            )
        elif model_type == "cnn":
            rmse, mae, _r2, n, best_ep = train_eval_cnn(
                params, inner_train, inner_val, info,
                nn_train_cfg, seed, do_early_stop=True
            )
        else:
            raise ValueError("Unknown model type: " + model_type)

        rmses.append(rmse)
        maes.append(mae)
        ns.append(n)
        if best_ep is not None:
            epochs.append(best_ep)

    w_rmse = weighted_rmse(rmses, ns)
    w_mae = weighted_mean(maes, ns)
    med_ep = int(np.median(np.asarray(epochs, dtype=int))) if len(epochs) > 0 else None
    return float(w_rmse), float(w_mae), med_ep

def outer_fit_eval(model_type, best_params, outer_train_areas, outer_test, info, nn_train_cfg, seed, fixed_epochs):
    if model_type in ["rf", "snn", "dnn"]:
        cfg2 = dict(nn_train_cfg)
        if model_type in ["snn", "dnn"]:
            cfg2["fixed_epochs"] = int(fixed_epochs)
        rmse, mae, r2, n, _ = train_eval_pixel_model(
            model_type, best_params, outer_train_areas, outer_test, info,
            cfg2, seed, do_early_stop=False if model_type != "rf" else False
        )
        # Build full prediction for spatial corr
        scaler = fit_scaler_pixel(outer_train_areas, info)
        m_te = info[outer_test]["mask"]
        Xte = scaler.transform(info[outer_test]["data"][m_te])

        if model_type == "rf":
            model = build_rf(best_params, seed)
            Xtr = np.vstack([scaler.transform(info[a]["data"][info[a]["mask"]]) for a in outer_train_areas])
            ytr = np.concatenate([info[a]["target"][info[a]["mask"]] for a in outer_train_areas])
            model.fit(Xtr, np.log1p(ytr))
            yp = np.expm1(model.predict(Xte))
        else:
            # re-train the NN once more for preds on outer_test in correct order
            # (already trained in train_eval_pixel_model; but we avoid returning the model object)
            # For spatial corr you can skip, but we compute it consistently:
            Xtr = np.vstack([scaler.transform(info[a]["data"][info[a]["mask"]]) for a in outer_train_areas])
            ytr = np.concatenate([info[a]["target"][info[a]["mask"]] for a in outer_train_areas])
            ytr_log = np.log1p(ytr)

            if model_type == "snn":
                model = build_snn(Xtr.shape[1], best_params)
            else:
                model = build_dnn(Xtr.shape[1], best_params)

            ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr_log)) \
                .shuffle(min(Xtr.shape[0], int(cfg2.get("shuffle_buf", 10000))), seed=seed, reshuffle_each_iteration=True) \
                .batch(int(cfg2.get("batch_size", 2048))).prefetch(tf.data.AUTOTUNE)

            model.fit(ds_tr, epochs=int(fixed_epochs), verbose=0)
            yp = np.expm1(model.predict(Xte, verbose=0).flatten())
            tf.keras.backend.clear_session()

        yp = np.nan_to_num(yp)
        pred_full = np.full(info[outer_test]["shape"], np.nan, dtype=np.float32)
        pred_full[m_te] = yp.astype(np.float32)
        spat = spatial_correlation_full(info[outer_test]["target"], pred_full, m_te)
        return rmse, mae, r2, n, spat["spatial_corr"]

    elif model_type == "cnn":
        cfg2 = dict(nn_train_cfg)
        cfg2["fixed_epochs"] = int(fixed_epochs)
        rmse, mae, r2, n, _ = train_eval_cnn(
            best_params, outer_train_areas, outer_test, info,
            cfg2, seed, do_early_stop=False
        )
        return rmse, mae, r2, n, float("nan")

    else:
        raise ValueError("Unknown model type: " + model_type)

def recommend_config(model_type, sel_df, outer_df):
    """
    Rule:
      most frequent selection across outer folds
      tie-break: lowest mean outer rmse among tied configs
    """
    counts = sel_df["cfg_key"].value_counts()
    top_count = int(counts.iloc[0])
    top_cfgs = [k for k, c in counts.items() if int(c) == top_count]

    if len(top_cfgs) == 1:
        chosen_key = top_cfgs[0]
    else:
        best_mean = None
        chosen_key = None
        for k in top_cfgs:
            mean_rmse = float(outer_df.loc[outer_df["cfg_key"] == k, "rmse"].mean())
            if (best_mean is None) or (mean_rmse < best_mean):
                best_mean = mean_rmse
                chosen_key = k

    # retrieve params of that key from sel_df (first occurrence)
    row = sel_df.loc[sel_df["cfg_key"] == chosen_key].iloc[0].to_dict()
    # keep only params fields (everything except outer_test + metrics)
    ignore = set(["outer_test", "inner_w_rmse", "inner_w_mae", "median_epoch", "cfg_key"])
    params = {k: row[k] for k in row.keys() if k not in ignore}

    return chosen_key, params, counts.to_dict()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # reproducibility
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    data_dir = cfg["data_dir"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    model_type = cfg["model_type"]  # rf / snn / dnn / cnn
    study_areas = cfg["study_areas"]
    bands = cfg["bands"]

    grids = cfg["grids"]
    if model_type not in grids:
        raise ValueError("No grid for model_type: " + model_type)

    n_random = cfg.get("n_random", None)
    nn_train_cfg = cfg.get("nn_training", {})

    # Load all data once per job
    print("Loading rasters...")
    info = {a: read_study_area_data(data_dir, a, bands) for a in study_areas}

    # Candidate sets for tuning
    candidates = sample_param_grid(grids[model_type], n_random=n_random, seed=seed)
    print("Model:", model_type, "| candidates:", len(candidates), "| n_random:", n_random)

    # Determine which outer fold to run (SLURM array)
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if slurm_id is not None:
        outer_id = int(slurm_id)
        if outer_id < 0 or outer_id >= len(study_areas):
            raise SystemExit("SLURM_ARRAY_TASK_ID out of range")
        outer_tests = [study_areas[outer_id]]
        print("Running ONE outer fold:", outer_tests[0])
    else:
        outer_tests = list(study_areas)
        print("Running ALL outer folds (single process).")

    outer_rows = []
    sel_rows = []

    for outer_test in outer_tests:
        print("\n===================================")
        print("OUTER TEST:", outer_test)
        print("===================================")

        outer_train_areas = [a for a in study_areas if a != outer_test]

        # INNER TUNING
        best = None
        best_params = None
        best_med_epoch = None

        for params in candidates:
            # ensure params numeric conversions for known keys
            p = deepcopy(params)

            # normalize possible YAML "null" strings
            if model_type == "rf":
                # keep None as None
                pass
            else:
                # keep numeric fields safe for NN/CNN (strings to float)
                for k in ["lr", "l2_dense", "l2_conv"]:
                    if k in p:
                        p[k] = safe_float(p[k])

            w_rmse, w_mae, med_ep = inner_score(model_type, p, outer_train_areas, info, nn_train_cfg, seed)

            if best is None or (w_rmse < best[0]) or (w_rmse == best[0] and w_mae < best[1]):
                best = (w_rmse, w_mae)
                best_params = p
                best_med_epoch = med_ep

        if best_params is None:
            raise RuntimeError("No best params found (inner tuning failed).")

        # OUTER TRAIN + EVAL
        fixed_epochs = best_med_epoch if best_med_epoch is not None else None
        if model_type in ["snn", "dnn", "cnn"]:
            if fixed_epochs is None:
                # fallback if something went weird
                fixed_epochs = int(nn_train_cfg.get("epochs", 50))
        else:
            fixed_epochs = None

        rmse, mae, r2, n_test, spatial_corr = outer_fit_eval(
            model_type, best_params, outer_train_areas, outer_test, info,
            nn_train_cfg, seed, fixed_epochs
        )

        key = cfg_key(model_type, best_params)

        sel_row = {"outer_test": outer_test}
        sel_row.update(best_params)
        sel_row.update({
            "inner_w_rmse": float(best[0]),
            "inner_w_mae": float(best[1]),
            "median_epoch": int(fixed_epochs) if fixed_epochs is not None else ""
        })
        sel_row["cfg_key"] = key

        out_row = {
            "area": outer_test,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "n_test": int(n_test),
            "spatial_corr": float(spatial_corr) if spatial_corr is not None else float("nan"),
            "cfg_key": key
        }
        for k, v in best_params.items():
            out_row["chosen_" + k] = v
        if fixed_epochs is not None:
            out_row["epochs"] = int(fixed_epochs)

        sel_rows.append(sel_row)
        outer_rows.append(out_row)

        pd.DataFrame([sel_row]).to_csv(os.path.join(out_dir, f"selection_{outer_test}.csv"), index=False)
        pd.DataFrame([out_row]).to_csv(os.path.join(out_dir, f"outer_{outer_test}.csv"), index=False)
        print("Saved selection_*.csv and outer_*.csv for", outer_test)


    if slurm_id is None:
        sel_df = pd.DataFrame(sel_rows)
        outer_df = pd.DataFrame(outer_rows)

        sel_df.to_csv(os.path.join(out_dir, "inner_selection.csv"), index=False)
        outer_df.to_csv(os.path.join(out_dir, "outer_results.csv"), index=False)

        nested_w_rmse = weighted_rmse(outer_df["rmse"].values, outer_df["n_test"].values)
        nested_w_mae = weighted_mean(outer_df["mae"].values, outer_df["n_test"].values)

        chosen_key, chosen_params, counts = recommend_config(model_type, sel_df, outer_df)

        rec = {
            "model": model_type,
            "recommended_config_key": chosen_key,
            "recommended_params": chosen_params,
            "selection_rule": "most_frequent_outer_selection (tie-break: lowest_mean_outer_rmse)",
            "nested_outer_weighted_rmse": float(nested_w_rmse),
            "nested_outer_weighted_mae": float(nested_w_mae),
            "selection_counts": counts,
            "n_random": int(n_random) if n_random is not None else None,
            "n_outer_folds_used": int(len(outer_df))
        }

        with open(os.path.join(out_dir, "recommended_config.json"), "w") as f:
            json.dump(rec, f, indent=2)

        print("\nNESTED OUTER SUMMARY")
        print("Weighted RMSE:", nested_w_rmse)
        print("Weighted MAE :", nested_w_mae)
        print("Recommended:", rec)

    print("\nDone.")

if __name__ == "__main__":
    main()
