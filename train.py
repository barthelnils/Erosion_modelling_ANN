#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final LOAO training script.

- Uses Leave-One-Area-Out (LOAO) validation
- Hyperparameters are taken from config.yaml 
- Saves one model + scaler per held-out area
- Writes per-fold metrics and overall mean metrics to CSV
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from modules.io_utils import read_study_area_data
from modules.patches import (
    expand_fields_individually,
    fit_scaler_for_cnn,
    apply_scaler,
    compute_safe_center_mask,
    extract_patches,
)
from modules.models import build_cnn, build_dnn, build_snn, build_rf
from modules.metrics import regression_metrics, classification_metrics, spatial_corr


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def loao_pairs(areas):
    """Generate (held_out, train_list) pairs for LOAO."""
    for a in areas:
        train = [x for x in areas if x != a]
        yield a, train


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------

def main(config_path: str = "config.yaml"):
    # 1) Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg["data_dir"]
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    # Study areas
    if cfg["study_areas"].get("folder_mode", False):
        areas = [
            os.path.splitext(f)[0]
            for f in os.listdir(data_dir)
            if f.endswith(".tif")
        ]
    else:
        areas = cfg["study_areas"]["list"]

    bands      = cfg["bands"]
    model_type = cfg["model_type"]              # 'cnn', 'dnn', 'snn', 'rf'
    params     = cfg["models"][model_type]      # final hyperparameters
    train_cfg  = cfg.get("training", {})

    batch_size = train_cfg.get("batch_size", 512)
    epochs     = train_cfg.get("epochs", 50)
    patience   = train_cfg.get("patience", 6)
    seed       = train_cfg.get("seed", 42)

    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    print("\n======================================")
    print(" Final LOAO training")
    print(" Model type:", model_type)
    print(" Areas:", areas)
    print("======================================\n")

    # Output subfolders
    models_root  = os.path.join(output_dir, "models", model_type)
    metrics_root = os.path.join(output_dir, "metrics")
    ensure_dir(models_root)
    ensure_dir(metrics_root)

    # 2) Load all study areas into memory
    print("Loading study areas...")
    info = {
        a: read_study_area_data(data_dir, a, bands)
        for a in areas
    }
    print("Loaded", len(info), "areas.")

    all_fold_metrics = []

    # -----------------------------------------------------------------
    # 3) LOAO loops over areas
    # -----------------------------------------------------------------
    for held_out, train_list in loao_pairs(areas):
        print("\n--------------------------------------")
        print(f"Held-out area: {held_out}")
        print("Training on:", train_list)
        print("--------------------------------------")

        # -----------------------------
        # CNN: patch-based training
        # -----------------------------
        if model_type == "cnn":
            patch_size = params["patch_size"]
            extra_expand = patch_size // 2 + 1

            # Expand training areas & fit scaler
            expanded_train = [
                expand_fields_individually(info[a], extra_expand)
                for a in train_list
            ]

            scaler = fit_scaler_for_cnn(expanded_train)

            # Build training patches
            Xtr_list, ytr_list = [], []
            for d in expanded_train:
                if not np.any(d["mask"]):
                    continue

                d_std = apply_scaler(d, scaler)
                center_mask = compute_safe_center_mask(d["mask"], patch_size)
                Xp, yp, _ = extract_patches(d_std, d["target"], center_mask, patch_size)
                if Xp.shape[0] == 0:
                    continue

                # drop pure-zero patches
                nonzero = ~np.all(Xp == 0.0, axis=(1, 2, 3))
                Xp = Xp[nonzero]
                yp = yp[nonzero]

                if Xp.shape[0] > 0:
                    Xtr_list.append(Xp)
                    ytr_list.append(yp)

            if len(Xtr_list) == 0:
                print(f"No training patches for held-out area {held_out}, skipping.")
                continue

            Xtr = np.concatenate(Xtr_list, axis=0)
            ytr = np.concatenate(ytr_list, axis=0)

            print("  Training patches:", Xtr.shape[0])

            # Validation data (expanded)
            d_val_exp = expand_fields_individually(info[held_out], extra_expand)
            if not np.any(d_val_exp["mask"]):
                print(f"No valid mask in expanded validation for {held_out}, skipping.")
                continue

            d_val_std = apply_scaler(d_val_exp, scaler)
            center_mask_v = compute_safe_center_mask(d_val_exp["mask"], patch_size)
            Xv, yv, idx = extract_patches(
                d_val_std, d_val_exp["target"], center_mask_v, patch_size
            )

            if Xv.shape[0] == 0:
                print(f"No validation patches in held-out area {held_out}, skipping.")
                continue

            print("  Validation patches:", Xv.shape[0])

            # Prepare datasets (log1p target)
            ytr_log = np.log1p(ytr)
            yv_log  = np.log1p(yv)

            ds_tr = (
                tf.data.Dataset
                .from_tensor_slices((Xtr, ytr_log))
                .shuffle(min(len(Xtr), 10000), seed=seed, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            ds_va = (
                tf.data.Dataset
                .from_tensor_slices((Xv, yv_log))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            # Build & train CNN
            model = build_cnn(
                input_shape=(patch_size, patch_size, len(bands)),
                p=params
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

            # Predict on validation patches
            y_pred_log = model.predict(Xv, verbose=0).flatten()
            y_pred = np.expm1(y_pred_log)
            y_pred = np.nan_to_num(y_pred)

            # Full expanded prediction grid
            H, W = d_val_exp["target"].shape
            pred_full_expanded = np.full((H, W), np.nan, dtype=np.float32)
            rr, cc = idx[:, 0], idx[:, 1]
            pred_full_expanded[rr, cc] = y_pred.astype(np.float32)

            # Map to original grid for evaluation
            d_orig = info[held_out]
            target_orig = d_orig["target"]
            mask_orig   = d_orig["mask"]

            pred_for_eval = pred_full_expanded
            valid_eval = mask_orig & np.isfinite(pred_for_eval)

            y_true_eval = target_orig[valid_eval]
            y_pred_eval = pred_for_eval[valid_eval]

            # Spatial corr uses full original grid
            spat = spatial_corr(target_orig, pred_for_eval, mask_orig)

        # -----------------------------
        # DNN / SNN / RF: per-pixel
        # -----------------------------
        else:
            # Fit scaler on training pixels
            mats = []
            for a in train_list:
                d = info[a]
                mats.append(d["data"][d["mask"]])
            Xall = np.vstack(mats)
            scaler = StandardScaler().fit(Xall)

            # Build training arrays
            Xtr_list, ytr_list = [], []
            for a in train_list:
                d = info[a]
                m = d["mask"]
                Xtr_list.append(scaler.transform(d["data"][m]))
                ytr_list.append(d["target"][m])

            Xtr = np.vstack(Xtr_list)
            ytr = np.concatenate(ytr_list)

            # Validation arrays
            d_val = info[held_out]
            m_val = d_val["mask"]
            Xv = scaler.transform(d_val["data"][m_val])
            yv = d_val["target"][m_val]

            # Build & train model
            if model_type == "rf":
                model = build_rf(params)
                # RF on log1p targets
                ytr_log = np.log1p(ytr)
                model.fit(Xtr, ytr_log)
                y_pred = np.expm1(model.predict(Xv))
                y_pred = np.nan_to_num(y_pred)

            else:
                input_dim = Xtr.shape[1]
                ytr_log = np.log1p(ytr)

                if model_type == "snn":
                    model = build_snn(input_dim, params)
                elif model_type == "dnn":
                    model = build_dnn(input_dim, params)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                es = EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True
                )

                ds_tr = (
                    tf.data.Dataset
                    .from_tensor_slices((Xtr, ytr_log))
                    .shuffle(min(len(Xtr), 10000), seed=seed, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )
                ds_va = (
                    tf.data.Dataset
                    .from_tensor_slices((Xv, np.log1p(yv)))
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                model.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=epochs,
                    callbacks=[es],
                    verbose=0
                )

                y_pred = np.expm1(model.predict(Xv, verbose=0).flatten())
                y_pred = np.nan_to_num(y_pred)

            # Build full prediction grid
            H, W = d_val["target"].shape
            pred_full = np.full((H, W), np.nan, dtype=np.float32)
            pred_full[m_val] = y_pred.astype(np.float32)

            target_full = d_val["target"]
            mask_orig   = d_val["mask"]

            valid_eval = mask_orig & np.isfinite(pred_full)
            y_true_eval = target_full[valid_eval]
            y_pred_eval = pred_full[valid_eval]

            spat = spatial_corr(target_full, pred_full, mask_orig)

        # 4) Compute metrics
        reg = regression_metrics(y_true_eval, y_pred_eval)
        cls = classification_metrics(y_true_eval, y_pred_eval)

        fold_metrics = {}
        fold_metrics.update(reg)
        fold_metrics.update(cls)
        fold_metrics.update(spat)
        fold_metrics["area"] = held_out
        fold_metrics["n_val"] = int(len(y_true_eval))

        all_fold_metrics.append(fold_metrics)

        print("  Fold metrics:", fold_metrics)

        # 5) Save model + scaler for this fold
        fold_dir = os.path.join(models_root, f"holdout_{held_out}")
        ensure_dir(fold_dir)

        if model_type == "rf":
            with open(os.path.join(fold_dir, "model.pkl"), "wb") as fh:
                pickle.dump({"model": model, "scaler": scaler}, fh)
        else:
            model.save(os.path.join(fold_dir, "model.h5"))
            with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as fh:
                pickle.dump(scaler, fh)

        tf.keras.backend.clear_session()

    # -----------------------------------------------------------------
    # 6) Save metrics across all folds
    # -----------------------------------------------------------------
    df = pd.DataFrame(all_fold_metrics)
    metrics_csv = os.path.join(metrics_root, f"{model_type}_loao_metrics.csv")
    df.to_csv(metrics_csv, index=False)

    # mean metrics (weighted by n_val)
    if df["n_val"].sum() > 0:
        w = df["n_val"] / df["n_val"].sum()
        weighted_rmse = np.sqrt(np.sum(w * (df["rmse"] ** 2)))
    else:
        weighted_rmse = np.nan

    summary = {
        "model_type": model_type,
        "weighted_rmse": weighted_rmse,
        "mean_rmse": df["rmse"].mean(),
        "mean_mae": df["mae"].mean(),
        "mean_f1": df["f1"].mean(),
        "mean_spatial_corr": df["spatial_corr"].mean()
    }
    summary_csv = os.path.join(metrics_root, f"{model_type}_loao_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\n======================================")
    print("LOAO training finished.")
    print("Per-fold metrics:", metrics_csv)
    print("Summary metrics:", summary_csv)
    print("======================================\n")


if __name__ == "__main__":
    main()
