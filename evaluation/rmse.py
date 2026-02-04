import numpy as np

def rmse(pred, ref):
    return np.sqrt(np.mean((pred - ref) ** 2))
