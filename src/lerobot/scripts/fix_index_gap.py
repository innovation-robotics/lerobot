import pandas as pd
from pathlib import Path

# 1. Setup paths
data_dir = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000").expanduser()
parquet_files = sorted(data_dir.glob("file-*.parquet"))

print(f"Starting index alignment for {len(parquet_files)} files...")

current_global_idx = 0

for f in parquet_files:
    # Load the WHOLE file (we need to preserve all columns: actions, states, etc.)
    df = pd.read_parquet(f)
    
    # Calculate the number of rows in this file
    num_rows = len(df)
    
    # Create the new continuous index based on its position in the whole dataset
    # This removes the "holes" (like the jump to 37611)
    new_indices = list(range(current_global_idx, current_global_idx + num_rows))
    
    df['index'] = new_indices
    
    # Overwrite the file with the fixed version
    df.to_parquet(f, index=False)
    
    print(f"✅ Fixed {f.name}: Rows {new_indices[0]} to {new_indices[-1]}")
    
    # Update the starting point for the next file
    current_global_idx += num_rows

print(f"\nAlignment Complete. Total frames re-indexed: {current_global_idx}")