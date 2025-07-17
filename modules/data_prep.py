# modules/data_prep.py
import os
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler

# Base data directory (override at runtime)
DATA_DIR = os.getenv('DATA_DIR', 'data')

def read_study_area_data(area, bands):
    """
    Load predictor bands and the target band from a GeoTIFF.

    Returns:
      data: 3D array (rows, cols, n_bands)
      target: 2D array (rows, cols)
      mask: boolean mask where target is valid (>=0 and != nodata)
      profile: rasterio profile for writing outputs
    """
    path = os.path.join(DATA_DIR, f"{area}.tif")
    with rasterio.open(path) as src:
        # read predictors
        arrs = [src.read(i)
                for i, desc in enumerate(src.descriptions, 1)
                if desc in bands]
        data = np.stack(arrs, axis=-1)

        # read target (assumed band 1)
        target = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999

        # mask out the declared nodata AND any negative values
        mask = (target != nodata) & (target >= 0)

        return data, target, mask, src.profile

def prepare_dataset(areas, bands):
    """
    Flatten and combine data across multiple study areas.

    Returns:
      info: dict area -> (shape, mask, profile)
      X: 2D array [n_samples, n_features]
      y: 1D array [n_samples]
    """
    info = {}
    X_list, y_list = [], []

    for area in areas:
        data, target, mask, profile = read_study_area_data(area, bands)
        info[area] = (data.shape, mask, profile)

        X_list.append(data[mask])
        y_list.append(target[mask])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return info, X, y

def normalize_features(X):
    """
    Standardize features to zero mean and unit variance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
