import data
import ddpm
from vae import vae

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance

import argparse

def fid_score(real_images, generated_images, device):
    real_images.to(torch.float64)
    generated_images.to(torch.float64)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_images = F.interpolate(real_images, size=(299,299), mode="bilinear", align_corners=False)
    fid.update(real_images, real=True)
    generated_images = F.interpolate(generated_images, size=(299,299), mode="bilinear", align_corners=False)
    fid.update(generated_images, real=False)
    score = fid.compute()
    return score

def evaluate(model, real_images, batch_size, device):
    n_samples = real_images.shape[0]
    samples = model.sample(n_samples, batch_size)
    score = fid_score(real_images, samples, device)
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
        choices=["cifar10"],
        required=True,
        help="Which test dataset."
    )
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"./trained_models/{args.model}_{args.dataset}_model.pth", map_location=device)
    if args.model == "ddpm":
        model = ddpm.ddpm.DDPM(
            beta=checkpoint["beta"],
            channel_mult=checkpoint["channel_mult"],
            image_dim=checkpoint["image_dim"],
            base_channels=checkpoint["base_channels"],
            dropout=checkpoint["dropout"],
            resample_with_conv=checkpoint["resample_with_conv"],
        )
    elif args.model == "vae":
        model = vae.VAE(
            in_ch=checkpoint["in_ch"],
            in_dim=checkpoint["in_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    model.to(device)

    if args.dataset == "cifar10":
        dataset = data.CIFAR10_1()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_samples = 1000
    real_images = next(iter(DataLoader(dataset, batch_size=n_samples, shuffle=True)))
    evaluate(model, real_images, args.batch_size, device)
