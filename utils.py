import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_image(img, rescale_method="tanh"):
    if rescale_method == "tanh":
        img = torch.tanh(img)
    elif rescale_method == "clamp":
        img = torch.clamp(img, -1.0, 1.0)
    else:
        raise ValueError("Unsupported rescale method")
    img = (img + 1) / 2
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()
