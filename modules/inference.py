# modules/inference.py
import os
import pickle
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model as keras_load_model
from .data_prep import read_study_area_data, normalize_features


def save_raster(pred_array, profile, output_path):
    """
    Write a numpy array to a GeoTIFF using the given profile.
    """
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=-9999
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_array.astype(rasterio.float32), 1)


def run_inference(model_path, scaler_path, areas, bands, output_dir, model_type='cnn'):
    """
    Load a trained model and scaler, perform predictions, and save GeoTIFFs.

    Args:
        model_path (str): path to saved model (h5 for NN, .pkl for RF)
        scaler_path (str): path to saved StandardScaler (.pkl)
        areas (list): list of study area names
        bands (list): list of band names
        output_dir (str): directory to write output .tif files
        model_type (str): one of ['cnn','dnn','snn','rf']
    """
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler: StandardScaler = pickle.load(f)

    # Load model
    if model_type == 'rf':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        is_nn = False
    else:
        model = keras_load_model(model_path)
        is_nn = True

    os.makedirs(output_dir, exist_ok=True)

    for area in areas:
        # Read raw data
        data, target, mask, profile = read_study_area_data(area, bands)
        # Flatten and scale
        X_flat = data[mask]
        X_scaled = scaler.transform(X_flat)

        # Reshape for NN models
        if is_nn:
            if model_type == 'cnn':
                X_in = X_scaled.reshape(-1, 1, 1, len(bands))
            else:
                X_in = X_scaled
            preds = model.predict(X_in).flatten()
        else:
            preds = model.predict(X_scaled)

        # Reconstruct full raster with nodata
        full = np.full(target.shape, -9999, dtype=np.float32)
        flat_full = full.flatten()
        flat_full[mask.flatten()] = preds
        full = flat_full.reshape(target.shape)

        # Save
        out_path = os.path.join(output_dir, f"{model_type}_predictions_{area}.tif")
        save_raster(full, profile, out_path)
