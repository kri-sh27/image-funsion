import torch

def predict(model, roi, sat):
    model.eval()
    with torch.no_grad():
        return model(roi, sat)
