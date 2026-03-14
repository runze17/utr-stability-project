import pandas as pd
import torch
from torch.utils.data import Dataset


class SalukiDataset(Dataset):
    """
    Dataset for mRNA stability prediction
    """

    def __init__(self, csv_path, tokenizer, max_length=4096):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sequence = row["sequence"]
        half_life = row["half_life"]

        utr5_len = row["utr5_len"]
        cds_len = row["cds_len"]
        utr3_len = row["utr3_len"]

        # tokenizer
        tokens = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # region mask
        region_mask = torch.zeros(self.max_length)

        end_utr5 = utr5_len
        end_cds = utr5_len + cds_len
        end_utr3 = utr5_len + cds_len + utr3_len

        region_mask[:end_utr5] = 0
        region_mask[end_utr5:end_cds] = 1
        region_mask[end_cds:end_utr3] = 2

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "region_mask": region_mask,
            "label": torch.tensor(half_life, dtype=torch.float)
        }