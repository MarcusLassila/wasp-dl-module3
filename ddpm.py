import data
import model
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
        ) # Predict noise
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2e-4)
        self.beta = torch.tensor(beta)
        self.T = self.beta.shape[0]
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

    def train(self, n_epochs):
        train_loader, val_loader = self.train_loader, self.val_loader
        model, optimizer, device, T = self.model, self.optimizer, self.device, self.T
        for epoch in range(n_epochs):
            model.train()
            for x in tqdm(train_loader, desc=f"Training epoch={epoch}"):
                x = x.to(device)
                t = torch.randint(low=0, high=T, size=(x.shape[0],)).to(device)
                eps = torch.randn(x.shape).to(device)
                alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
                z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                optimizer.zero_grad()
                noise_pred = model(z, t)
                loss = F.mse_loss(input=noise_pred, target=eps, reduction="sum")
                loss.backward()
                optimizer.step()

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
    x = next(iter(train_dataset))
    utils.plot_image(x)
    x = ddpm.sample()
    print(x.shape)
    utils.plot_image(x[0])
