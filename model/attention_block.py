import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        # multihead self-attention to capture contextual dependencies
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # convolution to project the input channels to the embedding dimension
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        # layer normalisation for stable training
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        # convolution and reshape to (batch, height*width, embed_dim)
        x = self.conv(x).reshape(batch, height * width, -1)
        # self-attention and add & normalise
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.layer_norm(attn_out + x)
        # reshape back to (batch, channels, height, width) and adjust channel order
        return attn_out.view(batch, -1, height, width).transpose(1, 3)
