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
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten the input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances to embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

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
    def __init__(self, image_channels=3, image_size=64, latent_size=256, hidden_dim=512, num_embeddings=512, commitment_cost=0.25, learning_rate=1e-3):
        super(ConvDeconvVQVAE, self).__init__()
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 4, 4)
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # (B, 256*4*4)
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings, hidden_dim, commitment_cost)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 4, 4)),  # (B, 256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # (B, image_channels, 64, 64)
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
vqvae = ConvDeconvVQVAE()
image = torch.randn((1, 3, 64, 64))  # Batch of one 64x64 RGB image
reconstructed_image, vq_loss = vqvae(image)
print(reconstructed_image.shape)  # Should output torch.Size([1, 3, 64, 64])
print(vq_loss)  # Vector quantization loss