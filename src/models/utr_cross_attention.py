import torch
import torch.nn as nn


class UTRCrossAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, region_mask):

        """
        hidden_states: [B, L, D]
        region_mask:   [B, L]
        """

        B, L, D = hidden_states.shape

        outputs = []

        for i in range(B):

            tokens = hidden_states[i]
            mask = region_mask[i]

            utr3_idx = mask == 2
            context_idx = mask != 2

            if utr3_idx.sum() == 0:
                outputs.append(tokens)
                continue

            Q = tokens[utr3_idx].unsqueeze(0)
            KV = tokens[context_idx].unsqueeze(0)

            attn_output, _ = self.attention(
                query=Q,
                key=KV,
                value=KV
            )

            tokens_new = tokens.clone()
            tokens_new[utr3_idx] = self.norm(
                tokens[utr3_idx] + attn_output.squeeze(0)
            )

            outputs.append(tokens_new)

        return torch.stack(outputs)