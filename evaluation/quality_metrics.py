import data
from ddpm import ddpm
from vae import vae

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.auto import tqdm

import argparse

def fid_score(model, dataloader, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    fid = fid.set_dtype(torch.float64)
    for real_samples in tqdm(dataloader, desc="Generating samples for FID"):
        batch_size = real_samples.shape[0]
        real_samples = real_samples.to(device)
        gen_samples = model.sample(batch_size)
        fid.update(real_samples, real=True)
        fid.update(gen_samples, real=False)
    print("Computing FID...")
    score = fid.compute()
    print(f"FID score: {score:.5f}")

def inception_score(model, n_samples, batch_size):
    is_metric = InceptionScore(normalize=True).to(device)
    n_batches, remainder = divmod(n_samples, batch_size)
    for _ in tqdm(range(n_batches), desc="Generating samples for inception score"):
        samples = model.sample(batch_size)
        is_metric.update(samples)
    samples = model.sample(remainder)
    is_metric.update(samples)
    print("Computing IS...")
    mean, std = is_metric.compute()
    print(f"Inception Score: {mean.item():.5f} ± {std.item():.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        choices=["fid", "is"],
        required=True,
        help="Which metric to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ddpm", "vae"],
        required=True,
        help="Which model to evaluate."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10-train", "cifar10-test", "cifar10.1"],
        default="cifar10-test",
        required=False,
        help="Which real example dataset to use for FID."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        required=False,
        help="Number of samples to evaluate (limited by len(dataset) in case of FID)."
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

    if args.metric == "is":
        inception_score(model, args.n_samples, args.batch_size)
    else: # FID
        if args.dataset == "cifar10-test":
            dataset = data.CIFAR10(train=False)
        elif args.dataset == "cifar10-train":
            dataset = data.CIFAR10(train=True)
        elif args.dataset == "cifar10.1":
            dataset = data.CIFAR10_1()
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        n_samples = min(args.n_samples, len(dataset))
        dataset = Subset(dataset, torch.arange(n_samples))
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        fid_score(model, data_loader, device)
