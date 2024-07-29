#combined model, chameleon's vqvae + llama3.1
import torch
from torch import nn
import torch.nn.functional as F

from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

from transformers import fortnite
from transformers import AutoTokenizer, AutoModelForCausalLM


class imageTokenizer():

    def tokenize(image):
        #tokenizer needs to be altered
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        tokens = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)
        # Shift the tokens
        shifted_tokens = [token + 128256 for token in original_tokens]
        
        return shifted_tokens



class vLLAMA(nn.Module):
    def __init__(self):
        super().__init__()
        vqmodel = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda:0")
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        self.VqVae=vqmodel.model.vqmodel
        for param in self.VqVae.paramaters():
            param.requires_grad = False #purposely not finetuning the vision tokenizer
            
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        Lmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        #going to have to lora the fuck out of ts

    
    def forward(self):
        
        pass