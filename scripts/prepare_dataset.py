import pandas as pd
from pathlib import Path


def build_dataset(input_path, output_path):
    """
    Convert Saluki feature table into training dataset
    """

    print("Loading Saluki dataset...")

    df = pd.read_csv(
        input_path,
        sep="\t"
    )

    print("Total transcripts:", len(df))

    # 拼接 full mRNA sequence
    df["sequence"] = df["5UTR"] + df["ORF"] + df["3UTR"]

    # 保存区域长度
    df["utr5_len"] = df["UTR5LEN"]
    df["cds_len"] = df["CDSLEN"]
    df["utr3_len"] = df["UTR3LEN"]

    # 只保留需要的列
    dataset = df[[
        "ENSID",
        "sequence",
        "utr5_len",
        "cds_len",
        "utr3_len",
        "HALFLIFE"
    ]]

    dataset.columns = [
        "gene_id",
        "sequence",
        "utr5_len",
        "cds_len",
        "utr3_len",
        "half_life"
    ]

    print("Saving dataset...")

    dataset.to_csv(
        output_path,
        index=False
    )

    print("Dataset saved:", output_path)
    print("Total samples:", len(dataset))


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data/raw/saluki/all_HLs_human_featTable.txt"

    output_file = project_root / "data/processed/saluki_dataset.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    build_dataset(input_file, output_file)