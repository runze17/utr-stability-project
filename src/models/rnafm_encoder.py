import torch
import torch.nn as nn
from transformers import AutoModel


class RNAFMEncoder(nn.Module):

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        return hidden_states