### WASP deep learning course, module 3

Implementation of denoising diffusion probabilistic model (DDPM) and a variational auto-encoder (VAE) for image generation.
Run scripts from the repo root folder since the paths are not absolute.

Run "source setup.sh" to add repo root to PYTHONPATH and install pyproject.

training_scripts contain scripts to train DDPM and VAE models, but the exact checkpoints used in the report can be downloaded from huggingface by the script "download_from_hf.py".

To evaluate FID or IS score, run evaluation/quality_metrics.py with suitable command line arguments (see source file for more info).