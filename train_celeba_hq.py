import data
import ddpm

import torch

torch.set_float32_matmul_precision('high')
train_dataset = data.CelebAHQ(train=True)
val_dataset = data.CelebAHQ(train=False)
beta = torch.linspace(start=1e-4, end=0.02, steps=1000)
channel_mult = (1, 1, 2, 2, 4, 4)
image_dim = train_dataset[0].shape
ddpm.DDPM(
    beta=beta,
    channel_mult=channel_mult,
    image_dim=image_dim,
    base_channels=128,
    dropout=0.0,
    resample_with_conv=True,
).train(train_dataset, val_dataset, batch_size=16, lr=2e-5, n_epochs=20, simul_batch_size=128)
