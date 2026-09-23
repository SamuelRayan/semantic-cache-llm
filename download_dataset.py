from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "ulab-ai/xRouteBench"
CONFIG_NAME = "llmrouter_generic"

output_dir = Path(r"C:\Users\HP\semantic-cache\data\raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME
)

print("\nDataset downloaded successfully!")
print(dataset)

for split_name, split_data in dataset.items():

    print(f"\nProcessing split: {split_name}")
    print(f"Rows: {len(split_data)}")
    print("Columns:", split_data.column_names)

    df = split_data.to_pandas()

    # Save as Parquet
    parquet_path = output_dir / f"xroutebench_{split_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    # Save as CSV
    csv_path = output_dir / f"xroutebench_{split_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV: {csv_path}")

print("\nAll files saved successfully!")