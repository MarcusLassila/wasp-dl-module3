import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class GaussianMLP(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lin_emb = nn.Linear(in_dim, hidden_dim)
        self.lin_mean = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.lin_emb(x)
        h = F.relu(h)
        mean = self.lin_mean(h)
        logvar = self.lin_logvar(h)
        cov = torch.diag_embed(logvar.exp())
        z = MultivariateNormal(mean, cov).rsample()
        return z, mean, logvar

class VAE(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = GaussianMLP(in_dim, hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        return self.decoder(z)

    def elbo(self, x):
        z, enc_mean, enc_logvar = self.encoder(x)
        kl_div = 0.5 * (1 + enc_logvar - enc_mean ** 2 - torch.exp(enc_logvar)).sum(dim=1)
        y = self.decode(z)
        log_likelihood = -F.binary_cross_entropy(input=y, target=x, reduction="sum")
        return (kl_div + log_likelihood).mean()
    
    @torch.no_grad()
    def generate(self, batch_size):
        device = next(self.decoder.parameters()).device
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        samples = self.decode(noise)
        return samples
