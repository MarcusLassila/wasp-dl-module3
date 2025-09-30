from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

Path("./trained_models").mkdir(parents=True, exist_ok=True)
filename = "vae_cifar10_model.pth"

_ = api.hf_hub_download(
    repo_id="Malassila/wasp-dl-module3-ddpm",
    filename=filename,
    local_dir="./trained_models",
    local_dir_use_symlinks=False,
)

print(f"Downloaded {filename} to ./trained_models")
