import data
from ddpm import ddpm
from vae import vae

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

import argparse

def fid_score(model, dataloader, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    fid = fid.set_dtype(torch.float64)
    for real_samples in tqdm(dataloader, desc="Generate samples"):
        batch_size = real_samples.shape[0]
        real_samples = real_samples.to(device)
        gen_samples = model.sample(batch_size)
        if model.__class__.__name__ == "DDPM":
            # Transform to [0,1]
            gen_samples = torch.clamp(gen_samples, -1.0, 1.0)
            gen_samples = (gen_samples + 1) / 2.0
        fid.update(real_samples, real=True)
        fid.update(gen_samples, real=False)
    print("Computing FID...")
    score = fid.compute()
    print(f"FID score: {score:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["ddpm", "vae"],
        required=True,
        help="Which model to evaluate."
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        choices=["cifar10", "cifar10.1"],
        required=True,
        help="Which test dataset."
    )
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(f"./trained_models/{args.model}_cifar10_model.pth", map_location=device)
    if args.model == "ddpm":
        model = ddpm.DDPM(
            beta=checkpoint["beta"],
            channel_mult=checkpoint["channel_mult"],
            image_dim=checkpoint["image_dim"],
            base_channels=checkpoint["base_channels"],
            dropout=checkpoint["dropout"],
            resample_with_conv=checkpoint["resample_with_conv"],
        )
        model.load(checkpoint["model_state_dict"])
    elif args.model == "vae":
        model = vae.VAE(
            in_ch=checkpoint["in_ch"],
            in_dim=checkpoint["in_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.dataset == "cifar10.1":
        dataset = data.CIFAR10_1()
    elif args.dataset == "cifar10":
        dataset = data.CIFAR10(train=False)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_samples = min(10000, len(dataset))
    dataset = Subset(dataset, torch.arange(n_samples))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    fid_score(model, data_loader, device)
