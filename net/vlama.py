#combined model, chameleon's vqvae + llama3.1
import torch
from torch import nn
import torch.nn.functional as F

from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

from transformers import fortnite
from transformers import AutoTokenizer, AutoModelForCausalLM


class imageEmbedder(nn.Module):
     def __init__(self):
        super().__init__()
        vqmodel = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda:0")
        for param in self.VqVae.paramaters():
            param.requires_grad = False #purposely not finetuning the vision tokenizer
        self.proj=nn.Linear(256,4096) #projection to embedding size of llama3.1
    def tokenize(self,tokens):
        
        # Shift the tokens
        shifted_tokens = [token + 128256 for token in original_tokens]
        
        return shifted_tokens
    def forward(self,image):
        encoded=vqmodel.encoder(image)
        encoded=vqmodel.quant_conv
        quantized,_,indices=vqmodel.quantize(encoded)
        tokens=self.tokenize(indices) 
        b, c, h, w = quantized.shape
        quantized= quantized.reshape(b, c, h*w)
        quantized= quantized.permute(0, 2, 1)
        embedding=self.proj(quantized) #returns tensor of shape (batch,seq_len,embed_dim), seq_len is the number of quantized patches returned from the convolutional encoder
        return embedding, tokens
        
        



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