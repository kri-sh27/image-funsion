import numpy as np
import torch

from pca.pca_fusion import pca_fusion
from cnn.model import FusionCNN
from cnn.train import train
from cnn.predict import predict
from ndvi.compute_ndvi import compute_ndvi
from evaluation.rmse import rmse
from evaluation.correlation import correlation


def load_dummy_data(H=128, W=128, C=4):
    """
    Temporary dummy data loader.
    Replace this with rasterio-based loading later.
    """
    uav_roi = np.random.rand(H, W, C)
    sat_roi = np.random.rand(H, W, C)
    sat_full = np.random.rand(H, W, C)
    gt_high_res = np.random.rand(H, W, C)

    return uav_roi, sat_roi, sat_full, gt_high_res


def main():
    print("🚀 Pipeline started")

    # ---------------------------------------------------
    # 1. Load UAV ROI + Satellite ROI
    # ---------------------------------------------------
    uav_roi, sat_roi, sat_full, gt_high_res = load_dummy_data()
    print("✅ Data loaded")

    # ---------------------------------------------------
    # 2. PCA-based ROI Fusion
    # ---------------------------------------------------
    fused_roi = pca_fusion(uav_roi, sat_roi)
    print("✅ PCA fusion completed")

    # ---------------------------------------------------
    # 3. Prepare data for CNN
    # ---------------------------------------------------
    fused_roi_t = torch.tensor(fused_roi).permute(2, 0, 1).unsqueeze(0).float()
    sat_full_t = torch.tensor(sat_full).permute(2, 0, 1).unsqueeze(0).float()
    gt_t = torch.tensor(gt_high_res).permute(2, 0, 1).unsqueeze(0).float()

    print("✅ Data converted to tensors")

    # ---------------------------------------------------
    # 4. CNN Training
    # ---------------------------------------------------
    model = FusionCNN(bands=fused_roi.shape[2])
    train(model, [(fused_roi_t, sat_full_t, gt_t)], epochs=5)
    print("✅ CNN training completed")

    # ---------------------------------------------------
    # 5. CNN Prediction (Full Scene)
    # ---------------------------------------------------
    pred_t = predict(model, fused_roi_t, sat_full_t)
    pred_img = pred_t.squeeze(0).permute(1, 2, 0).numpy()
    print("✅ High-resolution satellite prediction completed")

    # ---------------------------------------------------
    # 6. NDVI Computation
    # ---------------------------------------------------
    ndvi_pred = compute_ndvi(pred_img)
    ndvi_gt = compute_ndvi(gt_high_res)
    print("✅ NDVI computed")

    # ---------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------
    rmse_val = rmse(ndvi_pred, ndvi_gt)
    corr_val = correlation(ndvi_pred, ndvi_gt)

    print(f"📊 NDVI RMSE: {rmse_val:.4f}")
    print(f"📊 NDVI Correlation: {corr_val:.4f}")

    print("🎯 Pipeline completed successfully")


if __name__ == "__main__":
    main()
