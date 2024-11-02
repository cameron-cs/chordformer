import torch.nn as nn
from model.frequency_distributed_block import FrequencyDistributedBlock


class FeatureExtractionBlock(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        # frequency-distributed blocks to extract features from different frequency bands
        self.freq_block1 = FrequencyDistributedBlock(1, 64, num_bands)
        self.freq_block2 = FrequencyDistributedBlock(64, 128, num_bands)
        # max pooling to reduce the spatial dimensions
        self.pool = nn.MaxPool2d(2, ceil_mode=True, stride=2)

    def forward(self, x):
        # frequency-distributed blocks and pooling
        x = self.freq_block1(x)
        x = self.freq_block2(x)
        x = self.pool(x)
        return x
