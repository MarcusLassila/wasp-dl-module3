from ddpm import data, ddpm

import torch

torch.set_float32_matmul_precision('high')
dataset = data.CIFAR10()
beta = torch.linspace(start=1e-4, end=0.02, steps=1000)
channel_mult = (1, 2, 2, 2)
image_dim = dataset[0].shape
ddpm.DDPM(
    beta=beta,
    channel_mult=channel_mult,
    image_dim=image_dim,
    base_channels=128,
    dropout=0.1,
    resample_with_conv=True,
).train(dataset, batch_size=128, lr=2e-4, n_epochs=5000, simul_batch_size=512, grad_clip=1.0)
