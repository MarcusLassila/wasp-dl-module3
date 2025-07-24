import accelerate
import models
import utils

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

import time
from contextlib import nullcontext
from pathlib import Path

class DDPM:

    def __init__(self,
                 beta,
                 channel_mult,
                 train_dataset,
                 val_dataset,
                 batch_size,
                 device,
                 base_channels=128,
                 dropout=0.0,
                 lr=2e-4,
        ):
        self.input_dim = train_dataset[0].shape
        assert self.input_dim[1] == self.input_dim[2], "Only square images are supported"
        self.accelerator = accelerate.AcceleratorLite(batch_size=batch_size, do_compile=False)
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            self.autocast_context = nullcontext
        self.batch_size = batch_size
        self.device = device
        self.base_channels = base_channels
        self.dropout = dropout
        self.lr = lr
        self.model = models.UNet(
            size=self.input_dim[1],
            in_channels=self.input_dim[0],
            out_channels=self.input_dim[0],
            base_channels=base_channels,
            channel_mult=channel_mult,
            dropout=dropout,
            resample_with_conv=True,
        ) # Model that predict noise
        print(f"Using a U-net model with {utils.count_params(self.model)['n_params']:_} parameters")
        self.model, self.train_loader, self.val_loader = self.accelerator.prepare(self.model, train_dataset, val_dataset)
        if self.accelerator.running_ddp:
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.beta = beta.to(device)
        self.T = self.beta.shape[0]
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

    def train(self, n_epochs):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        step = 0
        best_val_loss = float('inf')
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            for x in self.train_loader:
                B = x.shape[0]
                t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                eps = torch.randn(x.shape).to(self.device)
                alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                self.optimizer.zero_grad()
                with self.autocast_context:
                    noise_pred = self.model(z, t)
                    loss = F.mse_loss(input=noise_pred, target=eps, reduction="mean")
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
                    f"| train loss: {loss.item():.6f} "
                    f"| grad norm: {norm:.2f} "
                    f"| images per sec: {im_per_sec:.1f} "
                    f"| dt: {t1 - t0:.2f}"
                )
                self.accelerator.print(log_msg, flush=True)
                t0 = time.time()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x in self.val_loader:
                    B = x.shape[0]
                    t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                    eps = torch.randn(x.shape).to(self.device)
                    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                    z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                    with self.autocast_context:
                        noise_pred = self.model(z, t)
                        loss = F.mse_loss(input=noise_pred, target=eps, reduction="mean")
                    val_loss += loss
                val_loss /= len(self.val_loader)
                if self.accelerator.running_ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                self.accelerator.print(f"epoch: {epoch} | val loss: {val_loss.item():.6f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    savepath = f"./trained_models/{self.train_loader.dataset_name}_model.pth"
                    Path("./trained_models").mkdir(parents=True, exist_ok=True)
                    torch.save(self.raw_model.state_dict(), savepath)

    def load(self, model_name):
        self.raw_model.load_state_dict(torch.load(model_name, map_location=self.device))
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
