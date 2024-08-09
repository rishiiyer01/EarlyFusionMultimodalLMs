#combined model, chameleon's vqvae + llama3.1
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

#loralayer to add to all linear layers in the LM
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).bfloat16())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank,dtype=torch.bfloat16) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim,dtype=torch.bfloat16))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

#class that takes in a linear layer and replaces with linear+LORA
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)



class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        chammodel = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
        for param in chammodel.parameters():
            param.requires_grad = False
        self.model = chammodel.model.vqmodel
        del chammodel
        for param in self.model.parameters():
            param.requires_grad = False  # purposely not finetuning the vision tokenizer
        vqvae_device = next(self.model.parameters()).device
        self.proj = nn.Linear(256, 4096, dtype=torch.bfloat16).to(vqvae_device)
        
        # Add image processor
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def tokenize(self, indices):
        # Shift the tokens
        shifted_tokens = [index + 128256 for index in indices.tolist()]
        return shifted_tokens

    def process_image(self, image):
        if isinstance(image, Image.Image):
            return self.image_processor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                return image.unsqueeze(0)
            return image
        else:
            raise ValueError("Image must be a PIL Image or a torch Tensor")

    def forward(self, image):
        image = self.process_image(image)
        encoded = self.model.encoder(image)
        encoded = self.model.quant_conv(encoded)
        quantized, _, indices = self.model.quantize(encoded)
        tokens = self.tokenize(indices) 
        b, c, h, w = quantized.shape
        quantized = quantized.reshape(b, c, h*w)
        quantized = quantized.permute(0, 2, 1)
        embedding = self.proj(quantized)
        return embedding, tokens
        
        



class VLLAMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_embedder = ImageEmbedder()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
        for param in self.llama.parameters():
            param.requires_grad = False #purposely not finetuning the original params

        
        lora_r = 8
        lora_alpha = 16
        for layer in self.llama.model.layers:
            #all linear modules in the repeated blocks
            layer.self_attn.q_proj=LinearWithLoRA(layer.self_attn.q_proj,lora_r,lora_alpha)
            layer.self_attn.k_proj=LinearWithLoRA(layer.self_attn.k_proj,lora_r,lora_alpha)
            layer.self_attn.v_proj=LinearWithLoRA(layer.self_attn.v_proj,lora_r,lora_alpha)
            layer.self_attn.o_proj=LinearWithLoRA(layer.self_attn.o_proj,lora_r,lora_alpha)
            layer.mlp.gate_proj=LinearWithLoRA(layer.mlp.gate_proj,lora_r,lora_alpha)
            layer.mlp.up_proj=LinearWithLoRA(layer.mlp.up_proj,lora_r,lora_alpha)
            layer.mlp.down_proj=LinearWithLoRA(layer.mlp.down_proj,lora_r,lora_alpha)
        self.llama.lm_head=LinearWithLoRA(self.llama.lm_head,lora_r,lora_alpha)
        
    def embedder(self, input_ids, images=None):
        
        text_embeddings = self.llama.get_input_embeddings()(input_ids)
        if images is None:
            combined_embeddings = text_embeddings
        else:
            image_embeddings, _ = self.image_embedder(images)
            # Concatenate image and text embeddings
            combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        image_tokens = image_embeddings.shape[1]
        return combined_embeddings, image_tokens

    def forward(self, input_ids, attention_mask, images=None, labels=None):
        combined_embeddings, image_tokens = self.embedder(input_ids, images)
        
        # Adjust attention_mask to include image tokens
        if images is not None:
            image_attention = torch.ones((combined_embeddings.shape[0], image_tokens), 
                                         dtype=attention_mask.dtype, 
                                         device=attention_mask.device)
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # Shift labels to align with text portion of output
        if labels is not None:
            shifted_labels = torch.full((labels.shape[0], combined_embeddings.shape[1]), 
                                        fill_value=-100, 
                                        dtype=labels.dtype, 
                                        device=labels.device)
            shifted_labels[:, -labels.shape[1]:] = labels
            labels = shifted_labels
    
        # Forward pass through LLAMA
        outputs = self.llama(inputs_embeds=combined_embeddings, 
                             attention_mask=attention_mask, 
                             labels=labels)
        
        return outputs




# Example usage for testing purposes
if __name__ == '__main__':
    model = VLLAMA().to("cuda")
    
    # Text-only input
    text_only = ["This is a text-only input."]
    with torch.no_grad():
        outputs_text = model(text_only)
    
    # Image-text input
    images = torch.randn(1, 3, 224, 224,dtype=torch.bfloat16).to("cuda:0")  # Assuming 224x224 image size
    text_with_image = ["Describe this image:"]
    with torch.no_grad():
        outputs_image_text = model(text_with_image, images)
    
    print("Text-only output shape:", outputs_text.logits.shape)
    print("Image-text output shape:", outputs_image_text.logits.shape)