import matplotlib.pyplot as plt
import torch
from pathlib import Path

def count_params(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'n_params': n_params, 'n_trainable_params': n_trainable}

def plot_image(img, rescale_method="tanh", name="temp_image"):
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
    plt.imshow(img)
    plt.title(rescale_method)
    Path("./images").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./images/{name}.png")

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