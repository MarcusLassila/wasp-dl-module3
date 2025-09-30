from vae import data, vae, train

import torch
from torch.utils.data import DataLoader

batch_size = 256
epochs = 1000
lr = 2e-4
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = data.CIFAR10(train=True)
test_dataset = data.CIFAR10(train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
channels, height, width = train_dataset[0].shape
assert height == width

model = vae.VAE(in_ch=channels, in_dim=height, latent_dim=latent_dim)
train.train_vae(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    epochs=epochs,
    device=device,
    lr=lr,
)
