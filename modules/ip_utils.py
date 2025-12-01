# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:28:10 2025

@author: barthe-n
"""

import os
import rasterio
import numpy as np

def read_study_area_data(data_dir, area, band_names):
    path = os.path.join(data_dir, area + ".tif")
    with rasterio.open(path) as src:
        desc = src.descriptions
        arrs = [
            src.read(i).astype(np.float32)
            for i, d in enumerate(desc, start=1) if d in band_names
        ]
        data = np.stack(arrs, axis=-1)
        target = src.read(1).astype(np.float32)

        nodata = src.nodata if src.nodata is not None else -9999
        feat_valid = np.all(np.isfinite(data), axis=-1)
        mask = (target != nodata) & (target >= 0) & feat_valid

        profile = src.profile

    return {"data": data, "target": target, "mask": mask, "profile": profile}


def save_raster(array, profile, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
