import torch
import torch.nn as nn

from src.models.rnafm_encoder import RNAFMEncoder
from src.models.utr_cross_attention import UTRCrossAttention
from src.models.utr_pooling import UTRPooling


class StabilityModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = RNAFMEncoder()

        hidden = self.encoder.hidden_dim

        self.cross_attention = UTRCrossAttention(hidden)

        self.utr_pooling = UTRPooling()

        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, sequences, region_mask):

        """
        sequences : list[str]
        region_mask : [B, L]
        """

        embeddings = self.encoder(sequences)

        x = self.cross_attention(
            embeddings,
            region_mask
        )

        pooled = self.utr_pooling(
            x,
            region_mask
        )

        out = self.mlp(pooled)

        return out.squeeze(-1)