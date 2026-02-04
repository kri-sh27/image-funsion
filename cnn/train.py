import torch
from cnn.model import FusionCNN

def train(model, dataloader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for roi, sat, gt in dataloader:
            pred = model(roi, sat)
            loss = loss_fn(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
