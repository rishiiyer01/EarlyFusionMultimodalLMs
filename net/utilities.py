import torch
from torch import nn
from torch.nn import functional as F
from vaePytorch import ConvDeconvVAE


class myLoss(nn.module):
    def __init__(self):
        super(myLoss,self).__init__()
    
    def forward(self,xhat,x):
        """
        Computes the reconstruction loss of the VAE.

        Args:
            x_hat: Reconstructed input data of shape (N, C, H, W).
            x: Input data for this timestep of shape (N, C, H, W).

        Returns:
            reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
        """
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")



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
        #we can actually save memory by separating out this loss function and calling backwards on each of them independently probably, but that can be implemented a lil later
        #for now this is probably fine
        return loss

