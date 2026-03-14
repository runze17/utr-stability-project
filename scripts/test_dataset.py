from transformers import AutoTokenizer
from src.data.saluki_dataset import SalukiDataset

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

dataset = SalukiDataset(
    "data/processed/saluki_dataset.csv",
    tokenizer
)

sample = dataset[0]

print("input_ids shape:", sample["input_ids"].shape)
print("region_mask shape:", sample["region_mask"].shape)
print("label:", sample["label"])