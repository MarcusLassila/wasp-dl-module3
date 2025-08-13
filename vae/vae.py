import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class GaussianMLP(nn.Module):
    
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.lin_emb = nn.Linear(in_features, hidden_units)
        self.lin_mean = nn.Linear(hidden_units, out_features)
        self.lin_logvar = nn.Linear(hidden_units, out_features)
        
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
        self.decoder = GaussianMLP(latent_dim, hidden_dim, in_dim)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        x, _, _ = self.decoder(z)
        return x

    def elbo(self, x):
        z, enc_mean, enc_logvar = self.encoder(x)
        kl_div = 0.5 * (1 + enc_logvar - enc_mean ** 2 - torch.exp(enc_logvar)).sum(dim=1)
        _, dec_mean, dec_logvar = self.decoder(z)
        dec_cov = torch.diag_embed(dec_logvar.exp()) 
        log_likelihood = MultivariateNormal(dec_mean, dec_cov).log_prob(x)
        return (kl_div + log_likelihood).mean()
    
    def generate(self, batch_size):
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        samples = self.decode(noise)
        return samples
