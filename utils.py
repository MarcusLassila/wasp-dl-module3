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
