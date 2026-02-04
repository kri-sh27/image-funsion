import numpy as np

def correlation(pred, ref):
    return np.corrcoef(pred.flatten(), ref.flatten())[0, 1]
