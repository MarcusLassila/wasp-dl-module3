import data
import ddpm

import torch

torch.set_float32_matmul_precision('high')
train_dataset = data.CelebAHQ(train=True)
val_dataset = data.CelebAHQ(train=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
beta = torch.linspace(start=1e-4, end=0.02, steps=1000)
channel_mult = (1, 1, 2, 2, 4, 4)
n_epochs = 10
ddpm.DDPM(
    beta=beta,
    channel_mult=channel_mult,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=batch_size,
    device=device,
    base_channels=128,
    dropout=0.0,
    lr=2e-5,
).train(n_epochs)
