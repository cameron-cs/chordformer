import torch
import torch.nn as nn


class ClassificationBlock(nn.Module):
    def __init__(self, embed_dim, num_chords, num_styles, guitar_types, guitar_statuses, dropout_rate):
        super().__init__()
        # embeddings for categorical features
        self.guitar_type_embedding = nn.Embedding(guitar_types, embed_dim)
        self.guitar_status_embedding = nn.Embedding(guitar_statuses, embed_dim)
        # dropout for regularisation
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layers for classification tasks
        self.fc_chord = nn.Linear(embed_dim * 3, num_chords)
        self.fc_style = nn.Linear(embed_dim * 3, num_styles)
        self.fc_status = nn.Linear(embed_dim * 3, guitar_statuses)
        self.fc_guitar_type = nn.Linear(embed_dim * 3, guitar_types)

    def forward(self, x, guitar_type, guitar_status):
        # embedding layers for guitar type and status
        guitar_type_embed = self.guitar_type_embedding(guitar_type)
        guitar_status_embed = self.guitar_status_embedding(guitar_status)

        # average pooling to reduce the feature dimension
        x = x.mean(dim=1)

        # concat features with embeddings and apply dropout
        x = torch.cat((x, guitar_type_embed, guitar_status_embed), dim=1)
        x = self.dropout(x)

        # out layers for each classification task
        chord_output = self.fc_chord(x)
        style_output = self.fc_style(x)
        status_output = self.fc_status(x)
        guitar_type_output = self.fc_guitar_type(x)

        return chord_output, style_output, status_output, guitar_type_output
