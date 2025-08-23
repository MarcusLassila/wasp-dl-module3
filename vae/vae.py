import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class EncoderMLP(nn.Module):

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
        return mean, logvar

class DecoderMLP(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class EncoderCNN(nn.Module):

    def __init__(self, in_channels, in_dim, latent_dim):
        super().__init__()
        self.conv_emb = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.hidden_dim = 64 * (in_dim // 4) ** 2
        self.lin_mean = nn.Linear(self.hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x):
        h = self.conv_emb(x)
        mean = self.lin_mean(h)
        logvar = self.lin_logvar(h)
        return mean, logvar

class DecoderCNN(nn.Module):

    def __init__(self, out_channels, latent_dim, hidden_dim):
        super().__init__()
        self.size = int((hidden_dim // 64) ** 0.5)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.latent_proj(x).view(x.shape[0], 64, self.size, self.size)
        return self.deconv(h)

class VAE(nn.Module):

    def __init__(self, in_channels, in_dim, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.encoder = EncoderCNN(
            in_channels=in_channels,
            in_dim=in_dim,
            latent_dim=latent_dim,
        )
        self.decoder = DecoderCNN(
            out_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dim=self.encoder.hidden_dim,
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return VAE.rsample(mean, logvar)
    
    def decode(self, z):
        return self.decoder(z)

    def elbo(self, x):
        enc_mean, enc_logvar = self.encoder(x)
        kl_div = -0.5 * torch.sum(1 + enc_logvar - enc_mean ** 2 - enc_logvar.exp())
        z = VAE.rsample(enc_mean, enc_logvar)
        y = self.decode(z)
        log_likelihood = -F.mse_loss(input=y, target=x, reduction="sum")
        return (-kl_div + log_likelihood) / x.shape[0]

    @torch.no_grad()
    def generate(self, batch_size):
        device = next(self.decoder.parameters()).device
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        samples = self.decode(noise)
        return samples

    @staticmethod
    def rsample(mean, logvar):
        cov = torch.diag_embed(logvar.exp())
        z = MultivariateNormal(mean, cov).rsample()
        return z
