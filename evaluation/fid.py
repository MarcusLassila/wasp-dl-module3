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

def sample_transformation(model_name, gen_samples):
    if model_name == "DDPM":
        gen_samples = torch.clamp(gen_samples, -1.0, 1.0)
        gen_samples = (gen_samples + 1) / 2.0
    return gen_samples

def fid_score(model, data_loader, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    total_samples = 0
    for real_samples in tqdm(data_loader, desc="Generate samples"):
        batch_size = real_samples.shape[0]
        total_samples += batch_size
        real_samples = real_samples.to(torch.float64).to(device)
        real_samples = F.interpolate(real_samples, size=(299,299), mode="bilinear", align_corners=False)
        gen_samples = model.sample(batch_size)
        gen_samples = sample_transformation(model.__class__.__name__, gen_samples)
        gen_samples = F.interpolate(gen_samples, size=(299,299), mode="bilinear", align_corners=False)
        fid.update(real_samples, real=True)
        fid.update(gen_samples, real=False)
    print("Computing FID...")
    score = fid.compute()
    print(f"FID score: {score:.5f}")
    print(f"Total samples: {total_samples}")

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
        model = ddpm.DDPM(
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
        model.to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.dataset == "cifar10":
        dataset = data.CIFAR10_1()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_samples = 2000
    dataset = Subset(dataset, torch.arange(n_samples))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    fid_score(model, data_loader, device)
