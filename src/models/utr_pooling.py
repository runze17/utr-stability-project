import torch
import torch.nn as nn


class UTRPooling(nn.Module):
    """
    Pool only 3'UTR tokens.
    region_mask:
        0 = 5UTR
        1 = CDS
        2 = 3UTR
    """

    def __init__(self, mode="mean"):
        super().__init__()
        assert mode in ["mean"]
        self.mode = mode

    def forward(self, hidden_states, region_mask):
        """
        hidden_states: [B, L, D]
        region_mask:   [B, L]
        returns:       [B, D]
        """
        pooled = []

        for i in range(hidden_states.size(0)):
            tokens = hidden_states[i]
            mask = region_mask[i]

            utr3_tokens = tokens[mask == 2]

            if utr3_tokens.size(0) == 0:
                # fallback: use global mean if no 3'UTR token is available
                pooled_vec = tokens.mean(dim=0)
            else:
                pooled_vec = utr3_tokens.mean(dim=0)

            pooled.append(pooled_vec)

        return torch.stack(pooled, dim=0)