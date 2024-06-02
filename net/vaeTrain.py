import torch
import torch
from torch import nn
from torch.nn import functional as F
from vaePytorch import ConvDeconvVQVAE

import torchvision

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

#dataloader needs to be defined
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model=ConvDeconvVQVAE()

num_epochs = 10
print(count_params(model)) 
for epoch in range(num_epochs):
    for images in dataloader:  
        model.optimizer.zero_grad()
        reconstructed_images, vq_loss = model(images)
        recon_loss = F.mse_loss(reconstructed_images, images)
        loss = recon_loss + vq_loss
        loss.backward()
        model.optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



