import pandas as pd

files = [
    "pairs.csv",
    "matchpairsDevTrain.csv",
    "matchpairsDevTest.csv",
    "mismatchpairsDevTrain.csv",
    "mismatchpairsDevTest.csv"
]

for file in files:

    path = "dataset/" + file

    df = pd.read_csv(path)

    print("\nFile:", file)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print(df.head())