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
        distances = torch.sum((flat_input.unsqueeze(1) - self.embeddings.weight) ** 2, dim=2)

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

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu'
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
    


class ViTVQVAE(nn.Module):
    def __init__(self, image_size=512, patch_size=32, in_channels=3, embed_dim=256, num_embeddings=8192, depth=6, num_heads=8, commitment_cost=0.25, learning_rate=1e-3):
        super(ViTVQVAE, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # Transformer Encoder
        self.encoder = ViTEncoder(embed_dim, depth, num_heads)

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embed_dim, commitment_cost)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 64, 64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 128, 128)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 256, 256)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),  # (B, in_channels, 512, 512)
            nn.Sigmoid(),  # To get pixel values between 0 and 1
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)

        # Encode
        patches = self.patch_embed(x)  # (B, n_patches, embed_dim)
        encoded_patches = self.encoder(patches)  # (B, n_patches, embed_dim)

        # Quantize
        quantized, vq_loss, _ = self.vq_layer(encoded_patches)  # (B, n_patches, embed_dim)

        # Reshape for decoder
        quantized = quantized.permute(0, 2, 1).contiguous()  # (B, embed_dim, n_patches)
        quantized = quantized.view(batch_size, self.embed_dim, self.image_size // self.patch_size, self.image_size // self.patch_size)  # (B, embed_dim, h, w)

        # Decode
        x_hat = self.decoder(quantized)  # (B, in_channels, 512, 512)
        return x_hat, vq_loss

    def reconstruct(self, x):
        x_hat, _ = self.forward(x)
        return x_hat

    def get_latent_codes(self, x):
        patches = self.patch_embed(x)  # (B, n_patches, embed_dim)
        encoded_patches = self.encoder(patches)  # (B, n_patches, embed_dim)
        _, _, encoding_indices = self.vq_layer(encoded_patches)
        return encoding_indices

# Example usage:
vit_vqvae = ViTVQVAE()
image = torch.randn((2, 3, 512, 512))  # Batch of 2 512x512 RGB images
reconstructed_image, vq_loss = vit_vqvae(image)
print(reconstructed_image.shape)  # Should output torch.Size([2, 3, 512, 512])
print(vq_loss)  # Vector quantization loss
# Example usage:
#vqvae = ConvDeconvVQVAE()
#image = torch.randn((2, 3, 32, 32))  # Batch of 2 64x64 RGB images
#reconstructed_image, vq_loss = vqvae(image)
#print(reconstructed_image.shape)  # Should output torch.Size([2, 3, 64, 64])
#print(vq_loss)  # Vector quantization loss