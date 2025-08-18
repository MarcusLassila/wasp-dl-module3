import data
import vae

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from math import prod
from pathlib import Path
import time

def train_vae(model, train_dataloader, val_dataloader, epochs, device, lr):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        acc_train_loss = 0.0
        avg_step_time = 0.0
        for x in train_dataloader:
            optimizer.zero_grad()
            B = x.shape[0]
            x = x.to(device)
            x = x.view(B, -1)
            loss = -model.elbo(x)
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
                B = x.shape[0]
                x = x.to(device)
                x = x.view(B, -1)
                loss = -model.elbo(x)
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
            "in_dim": model.in_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        torch.save(model_checkpoint, "./trained_models/VAE_model.pth")

if __name__ == "__main__":
    batch_size = 2048
    epochs = 500
    lr = 3e-4
    hidden_dim = 512
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = data.MNIST(train=True)
    test_dataset = data.MNIST(train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    in_dim = prod(train_dataset[0].shape)
    
    model = vae.VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    train_vae(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        epochs=epochs,
        device=device,
        lr=lr,
    )
