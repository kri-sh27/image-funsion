import numpy as np

def compute_ndvi(img, red_idx=2, nir_idx=3):
    red = img[:, :, red_idx].astype(float)
    nir = img[:, :, nir_idx].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi
