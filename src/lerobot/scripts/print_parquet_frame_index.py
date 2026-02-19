# import pandas as pd
# from pathlib import Path

# # Path to your first data parquet
# parquet_path = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000/file-000.parquet").expanduser()

# # Read just the first 5 rows
# df = pd.read_parquet(parquet_path, columns=["index", "timestamp"])
# print(df.head())

import pandas as pd
from pathlib import Path

# 1. Setup paths
dataset_path = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000").expanduser()
output_file = Path("dataset_heartbeat1.txt")

# 2. Find all parquet files in the data chunk
parquet_files = sorted(dataset_path.glob("file-*.parquet"))

print(f"Found {len(parquet_files)} data files. Extracting indices...")

all_data = []

# 3. Read specific columns from every file
for f in parquet_files:
    # We only load 'index' and 'timestamp' to keep it fast and memory-efficient
    df = pd.read_parquet(f, columns=["index", "timestamp"])
    all_data.append(df)

# 4. Combine and Export
full_df = pd.concat(all_data, ignore_index=True)

# index=False ensures we don't get a double-index in the text file
full_df.to_csv(output_file, sep='\t', index=False)

print(f"✅ Success! All {len(full_df)} frames exported to: {output_file}")
print(f"First 5 frames:\n{full_df.head()}")
print(f"Last 5 frames:\n{full_df.tail()}")