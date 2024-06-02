import torch
import torch
from torch import nn
from torch.nn import functional as F
from vaePytorch import ConvDeconvVAE



#dataloader needs to be defined




num_epochs = 10 
for epoch in range(num_epochs):
    for images in dataloader:  
        vqvae.optimizer.zero_grad()
        reconstructed_images, vq_loss = vqvae(images)
        recon_loss = F.mse_loss(reconstructed_images, images)
        loss = recon_loss + vq_loss
        loss.backward()
        vqvae.optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



