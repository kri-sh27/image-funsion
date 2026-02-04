import numpy as np

def spectral_angle_mapper(pred, ref):
    num = np.sum(pred * ref, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(ref, axis=-1)
    return np.mean(np.arccos(num / (den + 1e-6)))
