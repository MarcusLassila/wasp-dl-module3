import data
import vae

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from math import prod
from pathlib import Path
import time

def train_vae(model, train_dataloader, val_dataloader, epochs, device, lr):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        acc_train_loss = 0.0
        avg_step_time = 0.0
        for x in tqdm(train_dataloader, disable=device.type=="cuda", desc=f"epoch: {epoch}"):
            optimizer.zero_grad()
            x = x.to(device)
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            acc_train_loss += loss_item
            t1 = time.time()
            avg_step_time += t1 - t0
            t0 = time.time()
        train_loss.append(acc_train_loss / len(train_dataloader))
        avg_step_time /= len(train_dataloader)

        model.eval()
        acc_val_loss = 0.0
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(device)
                loss = model.loss(x)
                acc_val_loss += loss.item()
        val_loss.append(acc_val_loss / len(val_dataloader))

        log_msg = " | ".join([
            f"epoch: {epoch}/{epochs}",
            f"train loss: {train_loss[-1]:.6f}",
            f"val loss {val_loss[-1]:.6f}",
            f"step time: {avg_step_time:.4f}",
        ])
        print(log_msg, flush=True)
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "in_channels": model.in_channels,
            "in_dim": model.in_dim,
            "latent_dim": model.latent_dim,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        torch.save(model_checkpoint, "./trained_models/VAE_model.pth")

if __name__ == "__main__":
    batch_size = 128
    epochs = 5
    lr = 1e-3
    hidden_dim = 512
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = data.MNIST(train=True)
    test_dataset = data.MNIST(train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    channels, height, width = train_dataset[0].shape
    assert height == width
    
    model = vae.VAE(in_channels=channels, in_dim=height, latent_dim=latent_dim)
    train_vae(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        epochs=epochs,
        device=device,
        lr=lr,
    )
