import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent

csv_path = project_dir / "dataset" / "pairs.csv"

df = pd.read_csv(csv_path)

print("Column data types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nRows 2995 to 3005:")
print(df.iloc[2995:3005].to_string())

print("\nLast 5 rows:")
print(df.tail())

print("\nNon-null values in each column:")
print(df.notna().sum())