import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.saluki_dataset import SalukiDataset
from src.models.stability_model import StabilityPredictor


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    region_mask = torch.stack([x["region_mask"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "region_mask": region_mask,
        "labels": labels
    }


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

dataset = SalukiDataset(
    "data/processed/saluki_dataset.csv",
    tokenizer=tokenizer,
    max_length=512
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

batch = next(iter(loader))

model = StabilityPredictor(
    encoder_name="facebook/esm2_t6_8M_UR50D",
    hidden_dim=320,
    use_cross_attention=True
)

with torch.no_grad():
    preds = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        region_mask=batch["region_mask"]
    )

print("pred shape:", preds.shape)
print("pred:", preds)