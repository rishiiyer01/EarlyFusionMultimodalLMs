import torch
from torch import nn
import torch.nn.functional as F



# I tried my best to match the formatting for the VQVAE used in the Chameleon codebase
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
       


        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        
        z_q = z + (z_q - z).detach()
        
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, loss, min_encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, ch, ch_mult, num_res_blocks, attn_resolutions, resolution, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if i_level != self.num_resolutions - 1:
                blocks.append(nn.Conv2d(block_in, block_in, kernel_size=3, stride=2, padding=1))
                curr_res = curr_res // 2

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, out_channels, ch, ch_mult, num_res_blocks, attn_resolutions, resolution, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        blocks = []
        blocks.append(nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1))

        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if i_level != 0:
                blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                curr_res = curr_res * 2

        self.blocks = nn.Sequential(*blocks)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = z
        h = self.blocks(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,2,4), num_res_blocks=2, attn_resolutions=(16,), resolution=256, z_channels=256, num_embeddings=1024, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, ch, ch_mult, num_res_blocks, attn_resolutions, resolution, z_channels)
        self.quantize = VectorQuantizer(num_embeddings, z_channels, beta)
        self.decoder = Decoder(in_channels, ch, ch_mult, num_res_blocks, attn_resolutions, resolution, z_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, _ = self.quantize(z)
        x_recon = self.decoder(z_q)
        return x_recon, loss

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.quantize(z)
        return z_q

    def decode(self, z_q):
        x_recon = self.decoder(z_q)
        return x_recon

# Example usage
#vqvae = VQVAE()
#x = torch.randn(2, 3, 32, 32)
#x_recon, loss = vqvae(x)
#print(x_recon.shape, loss)

##encodedq = vqvae.encode(x)
#print(encodedq.shape)

#decoded = vqvae.decode(encodedq)
#print(decoded.shape)