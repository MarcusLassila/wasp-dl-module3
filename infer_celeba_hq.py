import ddpm
import utils

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("./trained_models/CelebAHQ_model.pth", map_location=device)
# ddpm_obj = ddpm.DDPM(
#     beta=checkpoint["beta"],
#     channel_mult=checkpoint["channel_mult"],
#     image_dim=checkpoint["image_dim"],
#     device=device,
#     base_channels=checkpoint["base_channels"],
#     dropout=checkpoint["dropout"],
#     resample_with_conv=checkpoint["resample_with_conv"],
# )
checkpoint = torch.load("./trained_models/CelebAHQ_model.pth", map_location=device)
ddpm_obj = ddpm.DDPM(
    beta=torch.linspace(start=1e-4, end=0.02, steps=1000),
    channel_mult=(1, 1, 2, 2, 4, 4),
    image_dim=(3,256,256),
    device=device,
    base_channels=128,
    dropout=0.0,
    resample_with_conv=True,
)
# ddpm_obj.load(checkpoint["state_dict"])
ddpm_obj.load(checkpoint)
batch_size = 1
gen_batch = ddpm_obj.sample(batch_size)
for i in range(batch_size):
    utils.plot_image(gen_batch[i], rescale_method="clamp", name=f"clamped_{i}")
for i in range(batch_size):
    utils.plot_image(gen_batch[i], rescale_method="tanh", name=f"tanhed_{i}")