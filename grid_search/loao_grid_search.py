#!/usr/bin/env python3
# -*- coding: ascii -*-

"""
LOAO grid search pipeline for soil erosion modelling.
ally.

Usage example (local):
    python loao_grid_search.py --config config.yaml

"""

import os
import sys
import math
import pickle
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import rasterio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.ensemble import RandomForestRegressor

from scipy.ndimage import binary_erosion, binary_dilation, label

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

import yaml

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------
# Utility: config + param grid
# ----------------------------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def generate_param_grid(grid_dict):
    """
    Given a dict like:
        {"lr":[1e-3,1e-4], "units":[64,128]}
    yield all combinations as dicts.
    """
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def select_param_combo(grid_dict):
    """
    Selects which param combination to run based on SLURM_ARRAY_TASK_ID.
    If SLURM_ARRAY_TASK_ID is unset, returns ALL combinations (for local runs).
    """
    all_combos = list(generate_param_grid(grid_dict))
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if slurm_id is None:
        # Local mode: run all
        return all_combos, None
    else:
        idx = int(slurm_id)
        if idx >= len(all_combos):
            print("SLURM_ARRAY_TASK_ID exceeds grid size, nothing to do.")
            sys.exit(0)
        return [all_combos[idx]], idx


# ----------------------------------------------------------------------
# Data I/O
# ----------------------------------------------------------------------

def read_study_area_data(data_dir, area, band_names):
    """
    Load predictors (bands) and target from a GeoTIFF.

    Returns:
      dict with keys: data, target, mask, shape, profile
    """
    path = os.path.join(data_dir, area + ".tif")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing raster: " + path)

    with rasterio.open(path) as src:
        desc = src.descriptions
        arrs = [
            src.read(i).astype(np.float32)
            for i, d in enumerate(desc, start=1) if d in band_names
        ]
        if len(arrs) == 0:
            raise RuntimeError("No matching bands in " + path)
        data = np.stack(arrs, axis=-1)

        target = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
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


def loao_areas(study_areas):
    """Yield (held_out, train_areas) for LOAO."""
    for a in study_areas:
        train = [x for x in study_areas if x != a]
        yield a, train


# ----------------------------------------------------------------------
# CNN-specific patch logic
# ----------------------------------------------------------------------

def expand_fields_individually(d, expand_pixels):
    """
    Per-field dilation expansion, then zero-fill new pixels,
    but keep the SAME array shape as original.
    """
    mask = d["mask"].astype(bool)
    labeled, n_fields = label(mask)
    expanded_mask = np.zeros_like(mask, dtype=bool)

    for i in range(1, n_fields + 1):
        expanded_mask |= binary_dilation(labeled == i, iterations=expand_pixels)

    new_zone = expanded_mask & ~mask

    data_exp = np.copy(d["data"])
    target_exp = np.copy(d["target"])

    data_exp[new_zone] = 0.0
    target_exp[new_zone] = 0.0

    d_new = d.copy()
    d_new["data"] = data_exp
    d_new["target"] = target_exp
    d_new["mask"] = expanded_mask
    d_new["shape"] = target_exp.shape

    return d_new


def fit_scaler_for_cnn(expanded_train_list):
    """
    Fit StandardScaler on ALL expanded training areas,
    but only on non-zero pixels.
    """
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
        raise RuntimeError("No valid pixels to fit scaler on for CNN.")
    X_all = np.vstack(arrs)
    return StandardScaler().fit(X_all)


def apply_scaler_to_data(d, scaler):
    H, W, C = d["data"].shape
    flat = d["data"].reshape(-1, C)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(H, W, C)


def compute_safe_center_mask(mask, patch_size):
    """
    Erode mask with a small kernel to define valid patch centers.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    radius = min(patch_size // 2, 2)
    size = 2 * radius + 1
    structure = np.ones((size, size), dtype=bool)
    return binary_erosion(mask, structure=structure, border_value=0)


def extract_patches(data_std, target, center_mask, patch_size):
    """
    Extract patches centered on pixels where center_mask is True.
    """
    pad = patch_size // 2
    data_std_pad = np.pad(
        data_std,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="reflect"
    )
    target_pad = np.pad(target, ((pad, pad), (pad, pad)), mode="reflect")

    idx = np.argwhere(center_mask)
    N = idx.shape[0]
    C = data_std.shape[-1]

    X = np.zeros((N, patch_size, patch_size, C), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)

    for k, (r, c) in enumerate(idx):
        X[k] = data_std_pad[r:r + patch_size, c:c + patch_size, :]
        y[k] = target_pad[r + pad, c + pad]

    return X, y, idx


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

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
        return {
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "accuracy": np.nan
        }
    cat_t = np.vectorize(categorize_soil_loss)(y_true)
    cat_p = np.vectorize(categorize_soil_loss)(y_pred)
    f1w = f1_score(cat_t, cat_p, average="weighted", zero_division=0)
    prec = precision_score(cat_t, cat_p, average="weighted", zero_division=0)
    rec = recall_score(cat_t, cat_p, average="weighted", zero_division=0)
    acc = accuracy_score(cat_t, cat_p)
    return {
        "f1": float(f1w),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc)
    }


def regression_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae)}


def spatial_correlation_full(y_true_full, y_pred_full, mask):
    valid = mask & np.isfinite(y_true_full) & np.isfinite(y_pred_full)
    if np.sum(valid) < 2:
        return {"spatial_corr": np.nan}
    corr = np.corrcoef(y_true_full[valid], y_pred_full[valid])[0, 1]
    return {"spatial_corr": float(corr)}


# ----------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------

def build_snn(input_dim, params):
    l2_dense = params["l2_dense"]
    units = params["units"]
    lr = params["lr"]
    L2r = regularizers.l2(l2_dense)
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units, activation="relu", kernel_regularizer=L2r),
        layers.Dense(1, activation="relu")
    ])
    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m


def build_dnn(input_dim, params):
    l2_dense = params["l2_dense"]
    layers_n = params["layers"]
    units = params["units"]
    dropout = params["dropout"]
    lr = params["lr"]
    L2r = regularizers.l2(l2_dense)
    m = models.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(layers_n):
        m.add(layers.Dense(units, activation="relu", kernel_regularizer=L2r))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="relu"))
    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m


def build_cnn(input_shape, params):
    lr = params["lr"]
    l2_conv = params["l2_conv"]
    l2_dense = params["l2_dense"]
    conv_layers_n = params["conv_layers"]
    filters = params["filters"]
    dropout = params["dropout"]
    kernel = params["kernel_size"]
    L2c = regularizers.l2(l2_conv)
    L2d = regularizers.l2(l2_dense)
    
    m = models.Sequential()
    m.add(layers.Input(shape=input_shape))

    # dynamic safe pooling
    safe_layers = 0
    h = input_shape[0]
    for _ in range(conv_layers_n):
        if h // 2 >= 2:
            safe_layers += 1
            h //= 2
        else:
            break

    for _ in range(safe_layers):
        m.add(layers.Conv2D(
            filters, (kernel, kernel),
            activation="relu",
            padding="same",
            kernel_regularizer=L2c
        ))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
        m.add(layers.MaxPooling2D((2, 2)))

    if safe_layers < conv_layers_n:
        m.add(layers.Conv2D(
            filters, (kernel, kernel),
            activation="relu",
            padding="same",
            kernel_regularizer=L2c
        ))
        if dropout > 0:
            m.add(layers.Dropout(dropout))

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(256, activation="relu", kernel_regularizer=L2d))
    if dropout > 0:
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="relu"))

    m.compile(optimizer=optimizers.Adam(lr), loss=tf.keras.losses.Huber())
    return m


def build_rf(params):
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1,
        random_state=42
    )


# ----------------------------------------------------------------------
# Core LOAO routines for each model type
# ----------------------------------------------------------------------

def run_loao_snn_dnn_rf(cfg, model_type, param_dict):
    """
    LOAO grid search for SNN, DNN, RF (per-pixel).
    """
    data_dir = cfg["data_dir"]
    base_results_dir = cfg["base_results_dir"]
    areas = cfg["study_areas"]
    bands = cfg["bands"]

    nn_cfg = cfg.get("nn_training", {})
    batch_size = nn_cfg.get("batch_size", 2048)
    epochs = nn_cfg.get("epochs", 50)
    patience = nn_cfg.get("patience", 5)
    seed = nn_cfg.get("seed", 42)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    job_suffix_parts = []
    for k, v in param_dict.items():
        job_suffix_parts.append(f"{k}{v}")
    job_suffix = "_".join(job_suffix_parts)

    results_dir = os.path.join(
        base_results_dir,
        f"{model_type}_loao_job_{job_suffix}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Read all areas once
    info = {a: read_study_area_data(data_dir, a, bands) for a in areas}
    all_metrics = []

    for held_out, train_areas in loao_areas(areas):
        print("Hold-out:", held_out)

        # Fit scaler on training areas
        train_infos = [info[a] for a in train_areas]
        mats = []
        for d in train_infos:
            X = d["data"][d["mask"]]
            mats.append(X)
        Xall = np.vstack(mats)
        scaler = StandardScaler().fit(Xall)

        # Build train data
        Xtr_list, ytr_list = [], []
        for a in train_areas:
            d = info[a]
            m = d["mask"]
            Xtr_list.append(scaler.transform(d["data"][m]))
            ytr_list.append(d["target"][m])
        Xtr = np.vstack(Xtr_list)
        ytr = np.concatenate(ytr_list)

        # Validation data
        d_val = info[held_out]
        m_val = d_val["mask"]
        Xv = scaler.transform(d_val["data"][m_val])
        yv = d_val["target"][m_val]

        # log1p transform
        ytr_log = np.log1p(ytr)

        if model_type == "rf":
            model = build_rf(param_dict)
            model.fit(Xtr, ytr_log)
            y_pred_log = model.predict(Xv)
            y_pred = np.expm1(y_pred_log)
        else:
            input_dim = Xtr.shape[1]
            if model_type == "snn":
                model = build_snn(input_dim, param_dict)
            else:
                model = build_dnn(input_dim, param_dict)

            es = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            )
            ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr_log))
            ds_tr = ds_tr.shuffle(
                min(Xtr.shape[0], 10000),
                seed=seed,
                reshuffle_each_iteration=True
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            ds_va = tf.data.Dataset.from_tensor_slices((Xv, np.log1p(yv)))
            ds_va = ds_va.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            model.fit(
                ds_tr,
                validation_data=ds_va,
                epochs=epochs,
                callbacks=[es],
                verbose=0
            )

            y_pred = np.expm1(model.predict(Xv, verbose=0).flatten())

        y_pred = np.nan_to_num(y_pred)

        # Build full prediction for spatial corr
        pred_full = np.full(d_val["shape"], np.nan, dtype=np.float32)
        pred_full[m_val] = y_pred.astype(np.float32)

        # Metrics
        reg_m = regression_metrics(yv, y_pred)
        cls_m = classification_metrics(yv, y_pred)
        spat_m = spatial_correlation_full(d_val["target"], pred_full, m_val)

        n_val = int(yv.size)
        m = {}
        m.update(reg_m)
        m.update(cls_m)
        m.update(spat_m)
        m["n_val"] = n_val
        m["area"] = held_out
        all_metrics.append(m)

        # Save model + scaler per fold
        if model_type == "rf":
            with open(os.path.join(results_dir, f"rf_{held_out}.pkl"), "wb") as fh:
                pickle.dump({"model": model, "scaler": scaler}, fh)
        else:
            model.save(os.path.join(results_dir, f"{model_type}_{held_out}.h5"))
            with open(os.path.join(results_dir, f"scaler_{held_out}.pkl"), "wb") as fh:
                pickle.dump(scaler, fh)

        tf.keras.backend.clear_session()

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(results_dir, "fold_metrics.csv"), index=False)

    if df["n_val"].sum() > 0:
        w = df["n_val"] / df["n_val"].sum()
        w_rmse = float(np.sqrt(np.sum(w * (df["rmse"] ** 2))))
    else:
        w_rmse = np.nan

    summary = {"model_type": model_type}
    summary.update(param_dict)
    summary["weighted_rmse"] = w_rmse
    summary["mean_f1"] = float(df["f1"].mean())
    summary["mean_corr"] = float(df["spatial_corr"].mean())
    return summary, results_dir


def run_loao_cnn(cfg, param_dict):
    """
    LOAO grid search for patch-based CNN.
    """
    data_dir = cfg["data_dir"]
    base_results_dir = cfg["base_results_dir"]
    areas = cfg["study_areas"]
    bands = cfg["bands"]

    nn_cfg = cfg.get("nn_training", {})
    batch_size = nn_cfg.get("batch_size", 512)
    epochs = nn_cfg.get("epochs", 50)
    patience = nn_cfg.get("patience", 5)
    seed = nn_cfg.get("seed", 42)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    patch_size = param_dict["patch_size"]
    extra_expand = patch_size // 2 + 1

    job_suffix_parts = []
    for k, v in param_dict.items():
        job_suffix_parts.append(f"{k}{v}")
    job_suffix = "_".join(job_suffix_parts)

    results_dir = os.path.join(
        base_results_dir,
        f"cnn_loao_job_{job_suffix}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # read original info
    info_orig = {a: read_study_area_data(data_dir, a, bands) for a in areas}
    all_metrics = []

    for held_out, train_areas in loao_areas(areas):
        print("\n============================")
        print("Hold-out area:", held_out)
        print("============================")

        # Expanded training
        expanded_train = [
            expand_fields_individually(info_orig[a], extra_expand)
            for a in train_areas
        ]
        scaler = fit_scaler_for_cnn(expanded_train)

        Xtr_list, ytr_list = [], []

        for d in expanded_train:
            if not np.any(d["mask"]):
                continue
            data_std = apply_scaler_to_data(d, scaler)
            safe_mask = compute_safe_center_mask(d["mask"], patch_size)
            Xp, yp, _ = extract_patches(data_std, d["target"], safe_mask, patch_size)
            if Xp.shape[0] == 0:
                continue
            nonzero = ~np.all(Xp == 0.0, axis=(1, 2, 3))
            Xp = Xp[nonzero]
            yp = yp[nonzero]
            if Xp.shape[0] > 0:
                Xtr_list.append(Xp)
                ytr_list.append(yp)

        if len(Xtr_list) == 0:
            print("No training patches for held out:", held_out)
            m_empty = {
                "area": held_out,
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "accuracy": np.nan,
                "spatial_corr": np.nan,
                "n_val": 0
            }
            all_metrics.append(m_empty)
            continue

        Xtr = np.concatenate(Xtr_list, axis=0)
        ytr = np.concatenate(ytr_list, axis=0)

        # Validation
        val_exp = expand_fields_individually(info_orig[held_out], extra_expand)
        if not np.any(val_exp["mask"]):
            print("No valid mask in expanded val for:", held_out)
            m_empty = {
                "area": held_out,
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "accuracy": np.nan,
                "spatial_corr": np.nan,
                "n_val": 0
            }
            all_metrics.append(m_empty)
            continue

        data_std_v = apply_scaler_to_data(val_exp, scaler)
        safe_mask_v = compute_safe_center_mask(val_exp["mask"], patch_size)
        Xv, yv, idx = extract_patches(
            data_std_v, val_exp["target"], safe_mask_v, patch_size
        )

        if Xv.shape[0] == 0:
            print("No validation patches for area:", held_out)
            m_empty = {
                "area": held_out,
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "accuracy": np.nan,
                "spatial_corr": np.nan,
                "n_val": 0
            }
            all_metrics.append(m_empty)
            continue

        # log1p targets
        ytr_log = np.log1p(ytr)
        yv_log = np.log1p(yv)

        ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr_log))
        ds_tr = ds_tr.shuffle(
            min(Xtr.shape[0], 10000),
            seed=seed,
            reshuffle_each_iteration=True
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        ds_va = tf.data.Dataset.from_tensor_slices((Xv, yv_log))
        ds_va = ds_va.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        model = build_cnn(
            (patch_size, patch_size, len(bands)),
            param_dict
        )
        es = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=epochs,
            callbacks=[es],
            verbose=0
        )

        y_pred_log = model.predict(Xv, verbose=0).flatten()
        y_pred = np.expm1(y_pred_log)
        y_pred = np.nan_to_num(y_pred)

        # Build full expanded grid of preds
        pred_full_expanded = np.full(val_exp["shape"], np.nan, dtype=np.float32)
        rr, cc = idx[:, 0], idx[:, 1]
        pred_full_expanded[rr, cc] = y_pred.astype(np.float32)

        # Map to original grid for evaluation
        d_orig = info_orig[held_out]
        gp = d_orig["target"]
        mask_orig = d_orig["mask"]

        pred_for_eval = np.array(pred_full_expanded, copy=True)
        valid_eval = mask_orig & np.isfinite(pred_for_eval)
        y_true_eval = gp[valid_eval]
        y_pred_eval = pred_for_eval[valid_eval]
        n_val = int(y_true_eval.size)

        reg_m = regression_metrics(y_true_eval, y_pred_eval)
        cls_m = classification_metrics(y_true_eval, y_pred_eval)
        spat_m = spatial_correlation_full(gp, pred_for_eval, mask_orig)

        m = {}
        m.update(reg_m)
        m.update(cls_m)
        m.update(spat_m)
        m["n_val"] = n_val
        m["area"] = held_out

        all_metrics.append(m)

        model.save(os.path.join(results_dir, f"cnn_{held_out}.h5"))
        with open(os.path.join(results_dir, f"scaler_{held_out}.pkl"), "wb") as fh:
            pickle.dump(scaler, fh)

        tf.keras.backend.clear_session()

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(results_dir, "fold_metrics.csv"), index=False)

    if df["n_val"].sum() > 0:
        w = df["n_val"] / df["n_val"].sum()
        w_rmse = float(np.sqrt(np.sum(w * (df["rmse"] ** 2))))
    else:
        w_rmse = np.nan

    summary = {"model_type": "cnn"}
    summary.update(param_dict)
    summary["weighted_rmse"] = w_rmse
    summary["mean_f1"] = float(df["f1"].mean())
    summary["mean_corr"] = float(df["spatial_corr"].mean())
    return summary, results_dir


# ----------------------------------------------------------------------
# Master summary writer
# ----------------------------------------------------------------------

def append_master_summary(base_results_dir, model_type, summary):
    summary_path = os.path.join(
        base_results_dir,
        f"{model_type}_loao_grid_master_summary.csv"
    )
    df = pd.DataFrame([summary])
    header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, "a") as f:
        df.to_csv(f, header=header, index=False)
    print("Summary appended:", summary)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to LOAO grid-search YAML config."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = cfg["data_dir"]
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    model_type = cfg["model_type"]
    grids = cfg["grids"]
    if model_type not in grids:
        raise ValueError("No grid defined for model_type: " + model_type)

    grid_dict = grids[model_type]
    combos, slurm_idx = select_param_combo(grid_dict)

    print("Model type:", model_type)
    print("Number of param combinations:", len(combos))
    if slurm_idx is not None:
        print("Running SLURM combo index:", slurm_idx)

    for param_dict in combos:
        print("\n==== New parameter set ====")
        print(param_dict)

        if model_type == "cnn":
            summary, _ = run_loao_cnn(cfg, param_dict)
        else:
            summary, _ = run_loao_snn_dnn_rf(cfg, model_type, param_dict)

        append_master_summary(base_results_dir, model_type, summary)


if __name__ == "__main__":
    main()
