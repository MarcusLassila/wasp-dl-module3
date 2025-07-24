import ddpm
import utils

import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_raw_data", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("./trained_models/CelebAHQ_model.pth", map_location=device)
ddpm_obj = ddpm.DDPM(
    beta=checkpoint["beta"],
    channel_mult=checkpoint["channel_mult"],
    image_dim=checkpoint["image_dim"],
    device=device,
    base_channels=checkpoint["base_channels"],
    dropout=checkpoint["dropout"],
    resample_with_conv=checkpoint["resample_with_conv"],
)
ddpm_obj.load(checkpoint["state_dict"])
batch_size = 16
gen_batch = ddpm_obj.sample(batch_size)
if args.save_raw_data:
    torch.save(gen_batch, "./images/image_batch.pth")
else:
    for i in range(batch_size):
        utils.plot_image(gen_batch[i], rescale_method="clamp", name=f"clamped_{i}")
    for i in range(batch_size):
        utils.plot_image(gen_batch[i], rescale_method="tanh", name=f"tanhed_{i}")
