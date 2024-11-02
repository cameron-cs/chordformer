import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention_block import AttentionBlock

class AttentionEncodingBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # attention block for capturing spatial relationships
        self.attention_block = AttentionBlock(128, embed_dim=embed_dim, num_heads=num_heads)
        # 3x3 convolution to further process features
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)
        # batch normalisation for stable training
        self.bn3 = nn.BatchNorm2d(embed_dim)
        # global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # positional encoding for Transformer input
        self.positional_encoding = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # transformer encoder for capturing sequence-level dependencies
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, x):
        # convolution, batch normalization, and activation
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        # positional encoding and pass through the Transformer encoder
        x = self.global_avg_pool(x).squeeze().unsqueeze(1) + self.positional_encoding
        x = self.transformer_encoder(x)
        return x
