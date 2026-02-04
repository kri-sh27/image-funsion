import numpy as np

def normalize(X):
    return X - np.mean(X, axis=0)

def reshape_to_pixels(img):
    H, W, C = img.shape
    return img.reshape(H * W, C), H, W
