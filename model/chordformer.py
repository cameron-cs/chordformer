import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention_encoding_block import AttentionEncodingBlock
from model.classification_block import ClassificationBlock
from model.feature_extraction_block import FeatureExtractionBlock


class Chordformer(nn.Module):
    def __init__(self, num_bands=4, embed_dim=128, num_heads=4, num_chords=8, num_styles=4, guitar_types=3,
                 guitar_statuses=2, dropout_rate=0.5):
        super().__init__()
        # feature extraction for initial audio processing
        self.feature_extraction = FeatureExtractionBlock(num_bands)

        # attention and encoding block for context understanding
        self.attention_encoding = AttentionEncodingBlock(embed_dim, num_heads)

        # classification block for predicting chords, styles, guitar type, and guitar status
        self.classification = ClassificationBlock(embed_dim, num_chords, num_styles, guitar_types, guitar_statuses,
                                                  dropout_rate)

    def forward(self, x, guitar_type, guitar_status):
        x = x.to(torch.float32).to(next(self.parameters()).device)
        guitar_type = guitar_type.to(torch.long).to(next(self.parameters()).device)
        guitar_status = guitar_status.to(torch.long).to(next(self.parameters()).device)

        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)

        # feature extraction from spectrogram
        x = self.feature_extraction(x)

        # attention-based encoding
        x = self.attention_encoding(x)

        # flatten or squeeze as needed for classification
        x = x.squeeze()
        chord_output, style_output, status_output, guitar_type_output = self.classification(x, guitar_type,
                                                                                            guitar_status)

        # log-softmax for all outputs
        return (
            F.log_softmax(chord_output, dim=1),
            F.log_softmax(style_output, dim=1),
            F.log_softmax(status_output, dim=1),
            F.log_softmax(guitar_type_output, dim=1)  # Added guitar type output
        )
