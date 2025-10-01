from ddpm import ddpm
from vae import vae

import torch
import time

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(f"./trained_models/ddpm_cifar10_model.pth", map_location=device)
ddpm_model = ddpm.DDPM(
    beta=checkpoint["beta"],
    channel_mult=checkpoint["channel_mult"],
    image_dim=checkpoint["image_dim"],
    base_channels=checkpoint["base_channels"],
    dropout=checkpoint["dropout"],
    resample_with_conv=checkpoint["resample_with_conv"],
)
ddpm_model.load(checkpoint["model_state_dict"])

checkpoint = torch.load(f"./trained_models/vae_cifar10_model_latest.pth", map_location=device)
vae_model = vae.VAE(
    in_ch=checkpoint["in_ch"],
    in_dim=checkpoint["in_dim"],
    latent_dim=checkpoint["latent_dim"],
)
vae_model.load_state_dict(checkpoint["model_state_dict"])
vae_model.to(device)

ddpm_sample_times = []
vae_sample_times = []
for _ in range(10):

    t0 = time.time()
    _ = ddpm_model.sample(batch_size)
    t1 = time.time()
    ddpm_sample_times.append(t1 - t0)
    
    t0 = time.time()
    _ = vae_model.sample(batch_size)
    t1 = time.time()
    vae_sample_times.append(t1 - t0)

ddpm_sample_times = torch.tensor(ddpm_sample_times)
ddpm_mean_sample_time = ddpm_sample_times.mean().item()
ddpm_std_sample_time = ddpm_sample_times.std().item()
vae_sample_times = torch.tensor(vae_sample_times)
vae_mean_sample_time = vae_sample_times.mean().item()
vae_std_sample_time = vae_sample_times.std().item()

print(f"ddpm sampling WCT: {ddpm_mean_sample_time:.5f} +- {ddpm_std_sample_time:.5f}")
print(f"vae sampling WCT: {vae_mean_sample_time:.5f} +- {vae_std_sample_time:.5f}")

