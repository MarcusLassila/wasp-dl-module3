from vae import data, vae, train

import torch
from torch.utils.data import DataLoader

batch_size = 2048
epochs = 300
lr = 1e-3
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = data.MNIST(train=True)
test_dataset = data.MNIST(train=False)
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
