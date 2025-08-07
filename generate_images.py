import ddpm
import utils

import torch
import argparse
from pathlib import Path

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
    for i in range(batch_size):
        utils.plot_image(gen_batch[i], rescale_method="clamp", name=f"clamped_{i}")
    for i in range(batch_size):
        utils.plot_image(gen_batch[i], rescale_method="tanh", name=f"tanhed_{i}")
