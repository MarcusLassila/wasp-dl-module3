from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

Path("./trained_models").mkdir(parents=True, exist_ok=True)
filename = "vae_mnist_model.pth"

_ = api.hf_hub_download(
    repo_id="Malassila/wasp-dl-module3-ddpm",
    filename=filename,
    local_dir="./trained_models",
)

print(f"Downloaded {filename} to ./trained_models")
