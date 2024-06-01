import torch
from torch import nn
from torch.nn import functional as F

class ConvDeconvVAE(nn.Module):
    def __init__(self, instrument_units, pitch_units, song_length, learning_rate, latent_size=256, hidden_dim=512):
        super(ConvDeconvVAE, self).__init__()
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.instrument_units = instrument_units
        self.pitch_units = pitch_units
        self.song_length = song_length

        input_shape = (self.pitch_units, self.song_length, self.instrument_units)
        flattened_dim = self.pitch_units * self.song_length * self.instrument_units

        if (instrument_units == 1):
            self.input_shape = (self.pitch_units, self.song_length)
            flattened_dim = self.pitch_units * self.song_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=8, stride=2, padding="same"),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, kernel_size=8, stride=(1, 2), padding="same"),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, kernel_size=4, padding="same"),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 1440),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1440),
            nn.Unflatten(1, (6, 80, 3)),
            nn.ConvTranspose2d(3, 12, kernel_size=8, stride=2, padding="same"),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.ConvTranspose2d(3, 12, kernel_size=8, stride=(1, 2), padding="same"),
            nn.LeakyReLU(0.2),
            nn.Sigmoid(),
            nn.ConvTranspose2d(3, instrument_units, kernel_size=4, padding="same"),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_size)
        self.logvar_layer = nn.Linear(hidden_dim, latent_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

    def get_latent_encoding(self, x):
        z = self.forward(x)[3]
        return z

    def predict(self, x):
        x_hat = self.forward(x)[0]
        return x_hat

    def bce_loss(self, x_hat, x):
        """
        Computes the reconstruction loss of the VAE.

        Args:
            x_hat: Reconstructed input data of shape (N, C, H, W).
            x: Input data for this timestep of shape (N, C, H, W).

        Returns:
            reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
        """
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
        return bce * x.size(1)  # Sum over all channels

    def loss_function(self, x_hat, x, mu, logvar):
        """
        Computes the negative variational lower bound loss term of the VAE.

        Args:
            x_hat: Reconstructed input data of shape (N, C, H, W).
            x: Input data for this timestep of shape (N, C, H, W).
            mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension.
            logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension.

        Returns:
            loss: Tensor containing the scalar loss for the negative variational lowerbound.
        """
        reconstruction_loss = self.bce_loss(x_hat, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_loss
        return loss / x.size(0)  # Average loss per sample

