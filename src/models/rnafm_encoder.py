import os
import torch
import torch.nn as nn
import fm
import urllib.request
from tqdm import tqdm


MODEL_URLS = [
    # HF 镜像（最快）
    "https://hf-mirror.com/ml4bio/RNA-FM/resolve/main/RNA-FM_pretrained.pth",

    # 官方源
    "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth",
]

CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")
MODEL_PATH = os.path.join(CACHE_DIR, "RNA-FM_pretrained.pth")


def download_with_progress(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))

        with open(save_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Downloading RNA-FM"
        ) as pbar:

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break

                f.write(chunk)
                pbar.update(len(chunk))


def ensure_model_downloaded():

    if os.path.exists(MODEL_PATH):
        print("RNA-FM checkpoint found locally.")
        return

    print("RNA-FM checkpoint not found. Downloading...")

    for url in MODEL_URLS:

        try:
            download_with_progress(url, MODEL_PATH)
            print("Download finished.")
            return
        except Exception as e:
            print(f"Download failed from {url}, trying next mirror...")

    raise RuntimeError("All download sources failed.")


class RNAFMEncoder(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()

        ensure_model_downloaded()

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()

        self.model = self.model.to(device)

        self.batch_converter = self.alphabet.get_batch_converter()

        self.hidden_dim = self.model.embed_dim

        self.device = device

    def forward(self, sequences):

        batch = [(str(i), seq) for i, seq in enumerate(sequences)]

        _, _, tokens = self.batch_converter(batch)

        tokens = tokens.to(self.device)

        outputs = self.model(tokens, repr_layers=[12])

        embeddings = outputs["representations"][12]

        return embeddings