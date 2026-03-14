import torch
import torch.nn as nn

from src.models.rnafm_encoder import RNAFMEncoder
from src.models.utr_cross_attention import UTRCrossAttention
from src.models.utr_pooling import UTRPooling


class StabilityPredictor(nn.Module):
    def __init__(
        self,
        encoder_name="facebook/esm2_t6_8M_UR50D",
        hidden_dim=320,
        num_heads=4,
        use_cross_attention=True,
    ):
        super().__init__()

        self.encoder = RNAFMEncoder(model_name=encoder_name)
        self.use_cross_attention = use_cross_attention

        if self.use_cross_attention:
            self.cross_attention = UTRCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )

        self.pooling = UTRPooling(mode="mean")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, region_mask):
        hidden_states = self.encoder(input_ids, attention_mask)

        if self.use_cross_attention:
            hidden_states = self.cross_attention(hidden_states, region_mask)

        pooled = self.pooling(hidden_states, region_mask)

        pred = self.mlp(pooled).squeeze(-1)

        return pred