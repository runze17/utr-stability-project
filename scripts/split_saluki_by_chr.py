import pandas as pd
from pathlib import Path


def normalize_chr(x):
    x = str(x).replace("chr", "").strip()
    return x


def main():
    project_root = Path(__file__).resolve().parents[1]

    saluki_path = project_root / "data/raw/saluki/all_HLs_human_featTable.txt"
    chr_path = project_root / "data/reference/gene_chr.tsv"

    output_dir = project_root / "data/processed/saluki_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Saluki data...")
    saluki = pd.read_csv(saluki_path, sep="\t")

    print("Loading chromosome mapping...")
    chr_df = pd.read_csv(chr_path, sep="\t")

    chr_df.columns = ["ENSID", "CHR"]
    chr_df["CHR"] = chr_df["CHR"].apply(normalize_chr)

    print("Merging...")
    df = saluki.merge(chr_df, on="ENSID", how="inner")

    # 构造 full mRNA sequence
    df["sequence"] = df["5UTR"] + df["ORF"] + df["3UTR"]

    # 只保留我们真正需要的字段
    df = df[[
        "ENSID",
        "CHR",
        "HALFLIFE",
        "UTR5LEN",
        "CDSLEN",
        "UTR3LEN",
        "5UTR",
        "ORF",
        "3UTR",
        "sequence"
    ]].copy()

    df.columns = [
        "gene_id",
        "chr",
        "half_life",
        "utr5_len",
        "cds_len",
        "utr3_len",
        "utr5",
        "orf",
        "utr3",
        "sequence"
    ]

    # Saluki-style chromosome split
    train_chr = {str(i) for i in range(1, 18)}   # 1-17
    val_chr = {"18", "19"}
    test_chr = {"20", "21", "22", "X", "Y"}

    train_df = df[df["chr"].isin(train_chr)].reset_index(drop=True)
    val_df = df[df["chr"].isin(val_chr)].reset_index(drop=True)
    test_df = df[df["chr"].isin(test_chr)].reset_index(drop=True)

    print(f"Total merged samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
    