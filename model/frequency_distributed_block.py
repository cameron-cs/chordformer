import torch
import torch.nn as nn


class FrequencyDistributedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_bands):
        super().__init__()
        self.num_bands = num_bands
        # local convolutional blocks for each frequency band
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(num_bands)
        ])
        # global convolution for the entire input
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # batch normalisation and activation for the global features
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # converting input to float32 for consistency
        x = x.to(torch.float32)
        # splitting the input into bands and apply local convolutions
        bands = torch.chunk(x, self.num_bands, dim=2)
        local_features = [conv(band) for conv, band in zip(self.local_convs, bands)]
        # global convolution and combine local and global features
        global_features = self.relu(self.bn(self.global_conv(x)))
        combined = torch.cat(local_features, dim=2) + global_features
        return combined
