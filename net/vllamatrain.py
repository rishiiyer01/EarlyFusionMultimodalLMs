##combined model
from tqdm import tqdm
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

import wandb

import random

def split_dataset(json_file, train_ratio=0.8):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * train_ratio)
    
    # Split the data
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data



class LLaVADataset(Dataset):
    def __init__(self, data, image_dir, tokenizer, image_embedder, max_length=2048):
        self.data = data
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_embedder = image_embedder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # Process text
        human_prompt = item['conversations'][0]['value']
        gpt_response = item['conversations'][1]['value']
        full_text = f"human: {human_prompt} gpt: {gpt_response}"
        
        # Tokenize text
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encodings = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Create labels
        labels = input_ids.clone()
        # Find the start of the "gpt:" token
        
        gpt_start = len(self.tokenizer.encode(f"human: {human_prompt} gpt: ", add_special_tokens=False))
        
        # Set labels before "gpt:" to -100
        labels[:gpt_start] = -100
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def custom_collate(batch):
    images = [item['image'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(torch.bfloat16)
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F

from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllama import VLLAMA


#HPS

num_epochs=5



import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler





run = wandb.init(
    # Set the project where this run will be logged
    project="EarlyFusionLM",
    # Track hyperparameters and run metadata
    config={
        "epochs": num_epochs,
    },
)







def train():
    device = torch.device("cuda")
    model = VLLAMA().to(device)
    # Split the data
    train_data, test_data = split_dataset('../../LLaVA-CC3M-Pretrain-595K/chat.json')

    # Create datasets
    train_dataset = LLaVADataset(
        data=train_data,
        image_dir='../../LLaVA-CC3M-Pretrain-595K',
        tokenizer=model.tokenizer,
        image_embedder=model.image_embedder
    )

    test_dataset = LLaVADataset(
        data=test_data,
        image_dir='../../LLaVA-CC3M-Pretrain-595K',
        tokenizer=model.tokenizer,
        image_embedder=model.image_embedder
    )

    train_dataloader = DataLoader(train_dataset, batch_size=3, collate_fn=custom_collate, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=3, collate_fn=custom_collate, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        for batch in tqdm(train_dataloader):
            # Move batch to the same device as the model
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Process images
            processed_images = torch.stack([model.image_embedder.process_image(img) for img in batch['image']]).to(torch.bfloat16).to(device)
            
            # Forward pass
            outputs = model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'], 
                            images=processed_images,
                            labels=batch['labels'])
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                processed_images = torch.stack([model.image_embedder.process_image(img) for img in batch['image']]).to(torch.bfloat16).to(device)
                
                outputs = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], 
                                images=processed_images,
                                labels=batch['labels'])
                
                test_loss += outputs.loss.item()
        
        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Test Loss: {avg_test_loss:.4f}")
            # Log metrics to wandb
        wandb.log({
            "train_loss": avg_loss,
            "val_loss": avg_test_loss,
            "epoch": epoch + 1
        })
    
        torch.save(model.module.state_dict(), 'vllama_trained.pth')
        




if __name__ == "__main__":
    train()