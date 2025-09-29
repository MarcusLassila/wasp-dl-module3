from ddpm import ddpm

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_images(images, rescale_method="clamp", name="temp_image"):
    # Create the 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        if rescale_method == "tanh":
            img = torch.tanh(img)
        elif rescale_method == "clamp":
            img = torch.clamp(img, -1.0, 1.0)
        elif rescale_method == "none":
            pass
        else:
            raise ValueError("Unsupported rescale method")
        img = (img + 1) / 2
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    Path("./images").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./images/{name}.png")
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--save-raw-data", action="store_true")
args = parser.parse_args()

checkpoint = torch.load(f"./trained_models/{args.model}.pth", map_location="cpu")
ddpm_obj = ddpm.DDPM(
    beta=checkpoint["beta"],
    channel_mult=checkpoint["channel_mult"],
    image_dim=checkpoint["image_dim"],
    base_channels=checkpoint["base_channels"],
    dropout=checkpoint["dropout"],
    resample_with_conv=checkpoint["resample_with_conv"],
)
ddpm_obj.load(checkpoint["model_state_dict"])
batch_size = args.batch_size
gen_batch = ddpm_obj.sample(batch_size)
if args.save_raw_data:
    Path("./images").mkdir(parents=True, exist_ok=True)
    torch.save(gen_batch, "./images/image_batch.pth")
else:
    plot_images(gen_batch, rescale_method="clamp", name=f"{args.model}_images")
