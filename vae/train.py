import data
import vae

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from math import prod
from pathlib import Path
import time

def train_vae(model, dataloader, epochs, device, lr):
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    step = 0
    total_steps = epochs * len(dataloader)
    train_loss = []
    for _ in range(epochs):
        t0 = time.time()
        for x in dataloader:
            optimizer.zero_grad()
            B = x.shape[0]
            x = x.to(device)
            x = x.view(B, -1)
            loss = -model.elbo(x)
            loss.backward()
            optimizer.step()
            step += 1
            loss_item = loss.item()
            train_loss.append(loss_item)
            t1 = time.time()
            if step % 20 == 0:
                images_per_sec = B / (t1 - t0)
                avg_loss = torch.tensor(train_loss[-10:]).mean().item()
                log_msg = f"step: {step}/{total_steps} | loss: {avg_loss:.6f} | images per sec: {images_per_sec:.1f} | batch time: {t1 - t0:.2f}"
                print(log_msg, flush=True)
            t0 = time.time()
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "in_dim": model.in_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "train_loss": train_loss,
        }
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        torch.save(model_checkpoint, "./trained_models/VAE_model.pth")

if __name__ == "__main__":
    batch_size = 2048
    epochs = 200
    lr = 3e-4
    hidden_dim = 512
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = data.MNIST()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    in_dim = prod(dataset[0].shape)
    
    model = vae.VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    train_vae(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        device=device,
        lr=lr,
    )
