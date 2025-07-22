import accelerate
import data
import models
import utils

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import time
from pathlib import Path

class DDPM:

    def __init__(self, beta, train_dataset, val_dataset, batch_size, device):
        self.accelerator = accelerate.AcceleratorLite(batch_size=batch_size)
        self.batch_size = batch_size
        self.input_dim = train_dataset[0].shape
        self.device = device
        self.model = models.UNet(
            size=32,
            in_channels=3,
            out_channels=3,
            base_channels=128,
            dropout=0.1,
            resample_with_conv=True,
        ) # Model that predict noise
        self.model, self.train_loader, self.val_loader = self.accelerator.prepare(self.model, train_dataset, val_dataset)
        self.beta = torch.tensor(beta).to(device)
        self.T = self.beta.shape[0]
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

    def train(self, n_epochs):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2e-4)
        step = 0
        best_val_loss = float('inf')
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            for x in self.train_loader:
                x = x.to(self.device)
                B = x.shape[0]
                t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                eps = torch.randn(x.shape).to(self.device)
                alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                self.optimizer.zero_grad()
                noise_pred = self.model(z, t)
                loss = F.mse_loss(input=noise_pred, target=eps, reduction="sum")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                loss = loss.detach()
                step += 1
                if self.accelerator.running_ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()
                im_per_sec = B * self.accelerator.world_size / (t1 - t0)
                log_msg = (
                    f"step: {step} "
                    f"| train loss: {loss.item() / B:.2f} "
                    f"| grad norm: {norm:.2f} "
                    f"| images per sec: {im_per_sec:.1f}"
                )
                self.accelerator.print(log_msg, flush=True)
                t0 = time.time()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x in self.val_loader:
                    x = x.to(self.device)
                    B = x.shape[0]
                    t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                    eps = torch.randn(x.shape).to(self.device)
                    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                    z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                    noise_pred = self.model(z, t)
                    loss = F.mse_loss(input=noise_pred, target=eps, reduction="sum")
                    val_loss += loss.item() / B
                val_loss /= len(self.val_loader)
                if self.accelerator.running_ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                self.accelerator.print(f"epoch: {epoch} | val loss: {val_loss:.2f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    savepath = f"./trained_models/{self.train_loader.dataset.__class__.__name__}_model.pth"
                    Path("./trained_models").mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), savepath)

    def load(self, model_name):
        self.model.load_state_dict(torch.load(model_name, map_location=self.device))
        self.model.eval()

    def sample(self):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(self.batch_size, *self.input_dim)
            for t in tqdm(range(self.T - 1, -1, -1), desc=f"Sampling {self.batch_size} images"):
                t = torch.tensor([t]).to(self.device)
                if t > 0:
                    z = torch.randn(self.batch_size, *self.input_dim)
                else:
                    z = torch.zeros(size=(self.batch_size, *self.input_dim))
                sigma_t = torch.sqrt((1 - self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t]))
                noise_pred = self.model(x, t)
                x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        return x

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    train_dataset = data.CIFAR10(train=True)
    val_dataset = data.CIFAR10(train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    ddpm = DDPM(
        beta=np.linspace(start=1e-4, stop=0.02, num=1000, dtype=np.float32),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        device=device,
    )
    ddpm.train(10)
    # ddpm.load("./trained_models/CIFAR10_model.pth")
    x = next(iter(train_dataset))
    utils.plot_image(x)
    x = ddpm.sample()
    for i in range(batch_size):
        utils.plot_image(x[i])
