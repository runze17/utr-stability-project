import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr

from transformers import AutoTokenizer

from src.data.saluki_dataset import SalukiDataset
from src.models.stability_model import StabilityModel


# -------------------------
# config
# -------------------------

TRAIN_CSV = "data/processed/saluki_split/train.csv"
VAL_CSV = "data/processed/saluki_split/val.csv"
TEST_CSV = "data/processed/saluki_split/test.csv"

BATCH_SIZE = 2
EPOCHS = 10
LR = 3e-5

CHECKPOINT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# evaluation
# -------------------------

def evaluate(model, loader):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            sequences = batch["sequence"]
            region_mask = batch["region_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            pred = model(sequences, region_mask)

            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    pearson = pearsonr(preds, labels)[0]

    mse = ((torch.tensor(preds) - torch.tensor(labels)) ** 2).mean()

    return mse.item(), pearson


# -------------------------
# training
# -------------------------

def train():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    train_dataset = SalukiDataset(TRAIN_CSV, tokenizer)
    val_dataset = SalukiDataset(VAL_CSV, tokenizer)
    test_dataset = SalukiDataset(TEST_CSV, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    model = StabilityModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        pbar = tqdm(train_loader)

        for batch in pbar:

            sequences = batch["sequence"]
            region_mask = batch["region_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            pred = model(sequences, region_mask)

            loss = criterion(pred, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch} Loss {loss.item():.4f}"
            )

        train_loss = total_loss / len(train_loader)

        val_mse, val_pearson = evaluate(model, val_loader)

        print(
            f"\nEpoch {epoch}"
            f"\nTrain Loss: {train_loss:.4f}"
            f"\nVal MSE: {val_mse:.4f}"
            f"\nVal Pearson: {val_pearson:.4f}\n"
        )

        if val_mse < best_val:

            best_val = val_mse

            torch.save(
                model.state_dict(),
                os.path.join(
                    CHECKPOINT_DIR,
                    "best_model.pt"
                )
            )

            print("Checkpoint saved.")

    print("\nTraining finished.")

    print("\nEvaluating on test set...")

    model.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pt")
        )
    )

    test_mse, test_pearson = evaluate(model, test_loader)

    print(
        f"\nTest MSE: {test_mse:.4f}"
        f"\nTest Pearson: {test_pearson:.4f}"
    )


# -------------------------
# main
# -------------------------

if __name__ == "__main__":

    train()