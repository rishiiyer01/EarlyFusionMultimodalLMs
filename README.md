# EarlyFusionMultimodalLMs
Experimenting with Early Fusion Multimodal LMs

-We started with training a VQ-VAE for image tokenization, with the plan of finetuning early fusion into Llama3 <br />
-Due to compute requirements, we pivoted to stitching an already trained VQ-VAE from Chameleon to Llama3.1 Instruct via finetuning on the liuhaotian/LLaVA-CC3M-Pretrain-595K dataset <br />
  -Training code available at net/vllamatrain.py <br />

The training script is currently complete and executable, the plan is to train on 4xH100. It's very likely that more data (and possibly compute) will be needed, since in traditional cross-attn style multimodality (like in llava), there is some text information within the CLIP embeddings. Do not hesistate to contact if you would like to contribute.
