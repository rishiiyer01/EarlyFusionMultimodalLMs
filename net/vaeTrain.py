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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

#dataloader needs to be defined
batch_size = 16
from PIL import Image
from torchvision.datasets import ImageFolder

from torch.utils.data import Dataset
from torchvision import transforms
import os
class ImageNetValidation(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return 0 as a dummy label
        return image, 0

# Define your transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Paths to your data
imagenet_path = "/datasets/imageNet/ILSVRC/Data/CLS-LOC"
train_dir = os.path.join(imagenet_path, "train")
val_dir = os.path.join(imagenet_path, "val")

# For training set
trainset = ImageFolder(train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# For validation set
testset = ImageNetValidation(val_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
model=VQVAE().to(device)
##hps for the graddescent

learning_rate = 0.001

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
        torch.save(model, f'/home/iyer.ris/vqvae/convarc_{epoch+1}.pth')



