import data
import model
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time

class DDPM:

    def __init__(self, beta, train_dataset, val_dataset, batch_size, device):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.input_dim = train_dataset[0].shape
        self.device = device
        self.model = model.UNet(
            size=32,
            in_channels=3,
            out_channels=3,
            base_channels=8,
            dropout=0.0,
            resample_with_conv=True,
        ) # Model that predict noise
        self.model.to(self.device)
        self.beta = torch.tensor(beta)
        self.T = self.beta.shape[0]
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

    def train(self, n_epochs):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2e-4)
        step = 0
        for epoch in range(n_epochs):
            self.model.train()
            t0 = time.time()
            for x in self.train_loader:
                x = x.to(self.device)
                t = torch.randint(low=0, high=self.T, size=(x.shape[0],)).to(self.device)
                eps = torch.randn(x.shape).to(self.device)
                alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1).to(self.device)
                z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                self.optimizer.zero_grad()
                noise_pred = self.model(z, t)
                loss = F.mse_loss(input=noise_pred, target=eps, reduction="sum")
                loss.backward()
                self.optimizer.step()
                step += 1
                t1 = time.time()
                im_per_sec = x.shape[0] / (t1 - t0)
                print(f"step: {step} | train loss: {loss.item():.5f} | images per sec: {im_per_sec:.1f}", flush=True)
                t0 = time.time()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x in self.val_loader:
                    x.to(self.device)
                    t = torch.randint(low=0, high=self.T, size=(x.shape[0],)).to(self.device)
                    eps = torch.randn(x.shape).to(self.device)
                    alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1).to(self.device)
                    z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                    noise_pred = self.model(z, t)
                    loss = F.mse_loss(input=noise_pred, target=eps, reduction="sum")
                    val_loss += loss.item()
                val_loss /= len(self.val_loader)
                print(f"epoch: {epoch} | val loss: {val_loss:.5f}", flush=True)
        
        torch.save(self.model.state_dict(), f"models/{self.train_loader.dataset.__class__.__name__}_final_model.pth")

    def load(self, model_name):
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

    def sample(self):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(self.batch_size, *self.input_dim)
            for t in range(self.T - 1, -1, -1):
                if t > 0:
                    z = torch.randn(self.batch_size, *self.input_dim)
                else:
                    z = torch.zeros(size=(self.batch_size, *self.input_dim))
                sigma_t = torch.sqrt((1 - self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t]))
                noise_pred = self.model(x, torch.tensor([t]))
                x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        return x

if __name__ == "__main__":
    train_dataset = data.CIFAR10(train=True)
    val_dataset = data.CIFAR10(train=False)
    ddpm = DDPM(
        beta=np.linspace(start=1e-4, stop=0.02, num=50, dtype=np.float32),
        train_dataset=val_dataset, # val is smaller
        val_dataset=val_dataset,
        batch_size=8,
        device=torch.device("cpu"),
    )
    ddpm.train(1)
    # ddpm.load("models/CIFAR10_final_model.pth")
    x = next(iter(train_dataset))
    utils.plot_image(x)
    x = ddpm.sample()
    utils.plot_image(x[0])
