import numpy as np
from sklearn.decomposition import PCA
from pca.pca_utils import normalize, reshape_to_pixels
def pca_fusion(uav_img, sat_img, k=4):
    X_uav, H, W = reshape_to_pixels(uav_img)
    X_sat, _, _ = reshape_to_pixels(sat_img)

    X_uav = normalize(X_uav)
    X_sat = normalize(X_sat)

    X = np.concatenate([X_uav, X_sat], axis=1)

    pca = PCA(n_components=k)
    Y = pca.fit_transform(X)
    X_fused = pca.inverse_transform(Y)

    X_fused = X_fused[:, :uav_img.shape[2]]
    return X_fused.reshape(H, W, uav_img.shape[2])
