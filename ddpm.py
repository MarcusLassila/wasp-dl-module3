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
                 image_dim,
                 base_channels=128,
                 dropout=0.0,
                 resample_with_conv=True,
                 do_compile=False,
        ):
        self.image_dim = image_dim
        assert self.image_dim[1] == self.image_dim[2], "Only square images are supported"
        self.accelerator = accelerate.AcceleratorLite(do_compile=do_compile)
        self.device = self.accelerator.device
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.T = beta.shape[0]
        self.beta = beta.to(self.device)
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.model = models.UNet(
            image_size=self.image_dim[1],
            in_channels=self.image_dim[0],
            out_channels=self.image_dim[0],
            base_channels=base_channels,
            channel_mult=channel_mult,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
        ).to(self.device) # Model that predict noise
        print(f"Using a U-net model with {utils.count_params(self.model)['n_params']:_} parameters")

    def train(self, train_dataset, val_dataset, batch_size, lr, n_epochs, simul_batch_size=64):
        input_dim = train_dataset[0].shape
        assert input_dim == self.image_dim

        assert simul_batch_size % batch_size == 0
        grad_accum_steps = simul_batch_size // batch_size

        if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_context = nullcontext()

        accelerator = self.accelerator
        model, train_loader, val_loader = accelerator.prepare(self.model, train_dataset, val_dataset, batch_size)
        if accelerator.running_ddp:
            raw_model = model.module
        else:
            raw_model = model

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        step = 0
        best_val_loss = float('inf')
        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            accum_train_loss = 0.0
            t0 = time.time()
            for i, x in enumerate(train_loader):
                final_grad_accum_step = (i + 1) % grad_accum_steps == 0
                B = x.shape[0]
                t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                eps = torch.randn(x.shape).to(self.device)
                alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                with autocast_context:
                    noise_pred = model(z, t)
                    loss = F.mse_loss(input=noise_pred, target=eps, reduction="mean")
                loss /= grad_accum_steps
                accum_train_loss += loss.detach()
                if accelerator.running_ddp:
                    model.require_backward_grad_sync = final_grad_accum_step
                loss.backward()
                if not final_grad_accum_step:
                    continue # Keep accumulating gradients

                # Gradient accumulation done, update parameters and log
                if accelerator.running_ddp:
                    dist.all_reduce(accum_train_loss, op=dist.ReduceOp.AVG)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize() # Syncronize processes before measuring t1
                t1 = time.time()
                step += 1
                im_per_sec = B * accelerator.world_size / (t1 - t0)
                log_msg = (
                    f"step: {step} "
                    f"| train loss: {accum_train_loss.item():.6f} "
                    f"| grad norm: {norm:.2f} "
                    f"| images per sec: {im_per_sec:.1f} "
                    f"| dt: {t1 - t0:.4f}"
                )
                accelerator.print(log_msg, flush=True)
                accum_train_loss = 0.0
                t0 = time.time()

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x in val_loader:
                    B = x.shape[0]
                    t = torch.randint(low=0, high=self.T, size=(B,)).to(self.device)
                    eps = torch.randn(x.shape).to(self.device)
                    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                    z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                    with autocast_context:
                        noise_pred = model(z, t)
                        loss = F.mse_loss(input=noise_pred, target=eps, reduction="mean")
                    val_loss += loss
                val_loss /= len(val_loader)
                if accelerator.running_ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                accelerator.print(f"epoch: {epoch} | val loss: {val_loss.item():.6f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    Path("./trained_models").mkdir(parents=True, exist_ok=True)
                    savepath = Path(f"./trained_models/{train_dataset.__class__.__name__}_model.pth")
                    checkpoint = {
                        "state_dict": raw_model.state_dict(),
                        "beta": self.beta,
                        "channel_mult": self.channel_mult,
                        "base_channels": self.base_channels,
                        "image_dim": self.image_dim,
                        "dropout": self.dropout,
                        "resample_with_conv": self.resample_with_conv,
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, savepath)

    def load(self, state_dict):
        state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def sample(self, batch_size):
        self.model.eval()
        x = torch.randn(batch_size, *self.image_dim).to(self.device)
        for t in tqdm(range(self.T - 1, -1, -1), desc=f"Sampling {batch_size} images"):
            if t > 0:
                z = torch.randn(batch_size, *self.image_dim).to(self.device)
            else:
                z = torch.zeros(size=(batch_size, *self.image_dim)).to(self.device)
            sigma_t = torch.sqrt(self.beta[t])
            noise_pred = self.model(x, t * torch.ones(batch_size, dtype=torch.long).to(self.device))
            x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        return x
