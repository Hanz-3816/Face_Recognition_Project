from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "lfw-deepfunneled"
PAIRS_CSV = PROJECT_DIR / "dataset" / "pairs.csv"


def load_lfw_pairs():
    pairs = pd.read_csv(PAIRS_CSV)

    records = []

    for index, row in pairs.iterrows():

        # First image
        name1 = str(row.iloc[0])
        number1 = int(row.iloc[1])

        # Genuine pair: fourth column is empty
        if pd.isna(row.iloc[3]):
            name2 = name1
            number2 = int(row.iloc[2])
            label = 1
            pair_type = "genuine"

        # Impostor pair: fourth column contains second image number
        else:
            name2 = str(row.iloc[2])
            number2 = int(row.iloc[3])
            label = 0
            pair_type = "impostor"

        # Construct image paths
        image1 = (
            DATASET_DIR / name1 /
            f"{name1}_{number1:04d}.jpg"
        )

        image2 = (
            DATASET_DIR / name2 /
            f"{name2}_{number2:04d}.jpg"
        )

        # Each consecutive group of 600 pairs is one fold
        fold = index // 600 + 1

        records.append({
            "img1_path": str(image1),
            "img2_path": str(image2),
            "label": label,
            "fold": fold,
            "pair_type": pair_type
        })

    return pd.DataFrame(records)


if __name__ == "__main__":

    data = load_lfw_pairs()

    print("\nDataset loaded successfully!")
    print("Total pairs:", len(data))

    print("\nPair distribution:")
    print(data["pair_type"].value_counts())

    print("\nFold distribution:")
    print(data.groupby(["fold", "pair_type"]).size())

    # Check whether all image paths exist
    missing = (
        ~data["img1_path"].map(lambda p: Path(p).exists())
        | ~data["img2_path"].map(lambda p: Path(p).exists())
    )

    print("\nMissing image pairs:", missing.sum())

    print("\nFirst five records:")
    print(data.head())