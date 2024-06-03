import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten the input except for the last dimension (embedding_dim)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances to embedding vectors using broadcasting
        embeddings_expanded = self.embeddings.weight.unsqueeze(0)  # Shape: (1, num_embeddings, embedding_dim)
        flat_input_expanded = flat_input.unsqueeze(1)  # Shape: (batch_size * num_latents, 1, embedding_dim)

        distances = torch.sum((flat_input_expanded - embeddings_expanded) ** 2, dim=2)  # Shape: (batch_size * num_latents, num_embeddings)


        # Get the closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        # Compute loss for embedding
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Preserve gradients
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices

class ConvDeconvVQVAE(nn.Module):
    def __init__(self, image_channels=3, image_size=32, latent_size=256, hidden_dim=256, num_embeddings=512, commitment_cost=0.25, learning_rate=1e-3):
        super(ConvDeconvVQVAE, self).__init__()
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 4, 4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 2, 2)
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # (B, 256*2*2)
            nn.Linear(256 * 4 , hidden_dim),
            nn.ReLU(inplace=True),
        )

        # VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings, hidden_dim, commitment_cost)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 2, 2)),  # (B, 256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # (B, image_channels, 32, 32)
            nn.Sigmoid(),  # To get pixel values between 0 and 1
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        latent = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(latent)
        x_hat = self.decoder(quantized)
        return x_hat, vq_loss

    def reconstruct(self, x):
        x_hat, _ = self.forward(x)
        return x_hat

    def get_latent_codes(self, x):
        latent = self.encoder(x)
        _, _, encoding_indices = self.vq_layer(latent)
        return encoding_indices

# Example usage:
#vqvae = ConvDeconvVQVAE()
#image = torch.randn((2, 3, 32, 32))  # Batch of 2 64x64 RGB images
#reconstructed_image, vq_loss = vqvae(image)
#print(reconstructed_image.shape)  # Should output torch.Size([2, 3, 64, 64])
#print(vq_loss)  # Vector quantization loss