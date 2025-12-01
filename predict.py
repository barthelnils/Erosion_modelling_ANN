#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOAO inference script.
- Loads the trained model for each held-out area
- Saves GeoTIFF predictions into output/predictions/<model_type>/
"""

import os
import yaml
import pickle
import numpy as np
import rasterio

from tensorflow.keras.models import load_model as keras_load_model

from modules.io_utils import read_study_area_data, save_raster
from modules.patches import (
    expand_fields_individually,
    fit_scaler_for_cnn,
    apply_scaler,
    compute_safe_center_mask,
    extract_patches
)


# ---------------------------------------------------------------------
# CNN patch-based inference
# ---------------------------------------------------------------------

def predict_cnn_for_area(area, info, model_dir, bands, patch_size, output_path):
    """
    CNN inference on a single study area using patch extraction.
    """

    # Load model + scaler
    model_path  = os.path.join(model_dir, "model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)

    model  = keras_load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load raw study area data
    d_orig = info[area]
    target = d_orig["target"]
    mask_orig = d_orig["mask"]
    profile = d_orig["profile"]

    # Expand fields
    extra_expand = patch_size // 2 
    d_exp = expand_fields_individually(d_orig, extra_expand)

    # Scale expanded data
    d_std = apply_scaler(d_exp, scaler)

    # Compute valid patch center mask
    center_mask = compute_safe_center_mask(d_exp["mask"], patch_size)

    # Extract patches
    Xp, _, idx = extract_patches(d_std, d_exp["target"], center_mask, patch_size)

    if Xp.shape[0] == 0:
        print(f"[WARN] No patches extracted for area {area}. Producing empty raster.")
        pred_full = np.full(target.shape, -9999, dtype=np.float32)
        save_raster(pred_full, profile, output_path)
        return

    # Predict
    y_pred_log = model.predict(Xp, verbose=0).flatten()
    y_pred     = np.expm1(y_pred_log)
    y_pred     = np.nan_to_num(y_pred)

    # Fill expanded grid
    H, W = target.shape
    pred_exp = np.full((H, W), np.nan, dtype=np.float32)
    rr, cc = idx[:, 0], idx[:, 1]
    pred_exp[rr, cc] = y_pred.astype(np.float32)

    # Map to original grid for saving (keep nodata where mask=false)
    pred_out = np.full((H, W), -9999, dtype=np.float32)
    valid = mask_orig & np.isfinite(pred_exp)
    pred_out[valid] = pred_exp[valid]

    # Save GeoTIFF
    save_raster(pred_out, profile, output_path)
    print(f"✓ Saved CNN prediction for {area} → {output_path}")


# ---------------------------------------------------------------------
# DNN / SNN / RF per-pixel inference
# ---------------------------------------------------------------------

def predict_pixel_model_for_area(area, info, model_dir, bands, output_path, model_type):
    """
    Pixel-based inference for DNN, SNN, RF.
    """

    model_path  = os.path.join(model_dir, "model.h5" if model_type != "rf" else "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Load model + scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if model_type == "rf":
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
            model = bundle["model"]
    else:
        model = keras_load_model(model_path)

    # Load raw data
    d = info[area]
    data   = d["data"]
    target = d["target"]
    mask   = d["mask"]
    profile = d["profile"]

    X = data[mask]
    Xs = scaler.transform(X)

    # Predict
    if model_type == "rf":
        preds = np.expm1(model.predict(Xs))
    else:
        preds = model.predict(Xs).flatten()

    preds = np.nan_to_num(preds)

    # Build full raster
    out = np.full(target.shape, -9999, dtype=np.float32)
    out_mask = out.flatten()
    out_mask[mask.flatten()] = preds
    out = out_mask.reshape(target.shape)

    save_raster(out, profile, output_path)
    print(f"✓ Saved {model_type.upper()} prediction for {area} → {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(config_path="config.yaml"):

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg["data_dir"]
    output_dir = cfg["output_dir"]
    model_type = cfg["model_type"]
    bands      = cfg["bands"]

    # Output folder for predictions
    pred_root = os.path.join(output_dir, "predictions", model_type)
    os.makedirs(pred_root, exist_ok=True)

    # Determine areas
    if cfg["study_areas"].get("folder_mode", False):
        areas = [
            os.path.splitext(f)[0]
            for f in os.listdir(data_dir)
            if f.endswith(".tif")
        ]
    else:
        areas = cfg["study_areas"]["list"]

    # Load study areas (tensor + mask + profile)
    info = {
        a: read_study_area_data(data_dir, a, bands)
        for a in areas
    }

    print("\n====================================")
    print("Running inference")
    print("Model type:", model_type)
    print("Areas:", areas)
    print("====================================\n")

    # -----------------------------------------------------------------
    # Loop over all areas
    # -----------------------------------------------------------------
    for area in areas:
        print(f"\n--- Predicting for {area} ---")

        # LOAO-trained model directory for this area
        model_dir = os.path.join(output_dir, "models", model_type, f"holdout_{area}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model for area '{area}' not found; expected in: {model_dir}"
            )

        output_path = os.path.join(pred_root, f"{area}.tif")

        if model_type == "cnn":
            patch_size = cfg["models"]["cnn"]["patch_size"]
            predict_cnn_for_area(area, info, model_dir, bands, patch_size, output_path)
        else:
            predict_pixel_model_for_area(
                area=area,
                info=info,
                model_dir=model_dir,
                bands=bands,
                output_path=output_path,
                model_type=model_type
            )

    print("\n✓ All predictions saved.")


if __name__ == "__main__":
    main()
