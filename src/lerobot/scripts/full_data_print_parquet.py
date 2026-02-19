import pandas as pd
from pathlib import Path

# 1. Setup paths
data_dir = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000").expanduser()
output_file = Path("full_dataset_inspect.txt")

# 2. Collect all parquet files
parquet_files = sorted(data_dir.glob("file-*.parquet"))
print(f"Reading {len(parquet_files)} files...")

all_dfs = []
for f in parquet_files:
    df = pd.read_parquet(f)
    all_dfs.append(df)

# 3. Merge everything
full_df = pd.concat(all_dfs, ignore_index=True)

# 4. Export to Tab-Delimited
# We use sep='\t' for tabs. 
# index=False because we want to use the 'index' column from the data, not pandas' internal row count.
full_df.to_csv(output_file, sep='\t', index=False)

print(f"✅ Success! Full inspection file created: {output_file}")
print(f"Total rows: {len(full_df)}")