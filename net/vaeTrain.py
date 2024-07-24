import torch

import wandb
from torch import nn
from torch.nn import functional as F
from vaePytorch import VQVAE
from functools import reduce
import torchvision
import torchvision.transforms as transforms
import operator
from timeit import default_timer
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using device: {device}")
print(f"Number of available GPUs: {num_gpus}")




def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

#dataloader needs to be defined
batch_size = 32
from datasets import load_dataset
ds=load_dataset("imagenet-1k")
train_dataset = ds['train']
validation_dataset = ds['validation']
from torch.utils.data import Dataset
from torchvision import transforms
import os


# Define your transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

num_workers=2*num_gpus
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')  # Ensure image is in RGB format
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Create PyTorch datasets
train_dataset = HFDataset(ds['train'], transform=transform)
test_dataset = HFDataset(ds['validation'], transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)


model=VQVAE()
if num_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
##hps for the graddescent

learning_rate = 0.001* num_gpus

epochs = 20
step_size =5
gamma = 0.5
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
num_epochs = 10
print(count_params(model)) 

#we probably should keep track of width and shit like that
run = wandb.init(
    # Set the project where this run will be logged
    project="EarlyFusionLM",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
    },
)



          
for epoch in range(num_epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0.0
    

    for images, _ in trainloader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_images, vq_loss = model(images)
        recon_loss = F.mse_loss(reconstructed_images, images)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(trainloader)
    
    scheduler.step()

    model.eval()
    val_loss = 0.0


    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            reconstructed_images, vq_loss = model(images)
            recon_loss = F.mse_loss(reconstructed_images, images)
            loss = recon_loss + vq_loss

            val_loss += loss.item()
            

    val_loss /= len(testloader)
   

    # Log metrics to wandb
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch + 1
    })

   # wandb.log({"reconstructed_images": [wandb.Image(reconstructed_images[i]) for i in range(min(len(reconstructed_images), 10))]})
    
    t2 = default_timer()
    print(f'Epoch {epoch+1}, Time: {t2 - t1:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    if (epoch + 1) % step_size == 0:
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), f'/home/iyer.ris/vqvae/convarc_{epoch+1}.pth')
        else:
            torch.save(model.state_dict(), f'/home/iyer.ris/vqvae/convarc_{epoch+1}.pth')


