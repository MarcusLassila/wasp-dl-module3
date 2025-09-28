import data
import ddpm
import vae

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

import argparse

def fid_score(real_images, generated_images, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_images = F.interpolate(real_images, size=(299,299), mode="bilinear", align_corners=False)
    fid.update(real_images, real=True)
    generated_images = F.interpolate(generated_images, size=(299,299), mode="bilinear", align_corners=False)
    fid.update(generated_images, real=False)
    score = fid.compute()
    return score

def evaluate(model, dataset, device, num_samples=100):
    assert num_samples <= len(dataset)
    samples = model.sample(num_samples)
    score = fid_score(dataset[:num_samples], samples, device)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"trained_models/{args.model}_{args.dataset}_model.pth", map_location=device)
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
        model = vae.vae.VAE(
            in_channels=checkpoint["in_channels"],
            in_dim=checkpoint["in_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    if args.dataset == "cifar10":
        dataset = data.CIFAR10_1()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    evaluate(model, dataset, device)
