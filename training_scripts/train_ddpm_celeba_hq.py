from ddpm import data, ddpm

import torch

torch.set_float32_matmul_precision('high')
dataset = data.CelebAHQ()
beta = torch.linspace(start=1e-4, end=0.02, steps=1000)
channel_mult = (1, 1, 2, 2, 4, 4)
image_dim = dataset[0].shape
ddpm.DDPM(
    beta=beta,
    channel_mult=channel_mult,
    image_dim=image_dim,
    base_channels=128,
    dropout=0.0,
    resample_with_conv=True,
).train(dataset, batch_size=16, lr=2e-5, n_epochs=1000, simul_batch_size=64, grad_clip=1.0)
