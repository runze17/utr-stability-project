import pandas as pd
import torch
from torch.utils.data import Dataset


class SalukiDataset(Dataset):
    """
    Dataset for mRNA stability prediction.

    Each sample returns:
        input_ids   : tokenized mRNA sequence
        region_mask : mask indicating 3'UTR region
        label       : mRNA half-life
    """

    def __init__(self, csv_path, tokenizer, max_len=4096):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = row["sequence"]
        label = torch.tensor(row["half_life"], dtype=torch.float32)

        tokenized = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)

        # 3'UTR region mask
        region_mask = torch.zeros(self.max_len, dtype=torch.float32)

        utr5_len = int(row["utr5_len"])
        cds_len = int(row["cds_len"])
        utr3_len = int(row["utr3_len"])

        start = utr5_len + cds_len
        end = start + utr3_len

        end = min(end, self.max_len)

        region_mask[start:end] = 1

        return {
            "input_ids": input_ids,
            "region_mask": region_mask,
            "label": label
        }