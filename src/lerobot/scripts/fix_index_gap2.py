import pandas as pd
from pathlib import Path

# 1. Setup paths
# We point to the data chunk where your Parquet files live
data_dir = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000").expanduser()
parquet_files = sorted(data_dir.glob("file-*.parquet"))

print(f"Starting Global Index alignment for {len(parquet_files)} files...")

current_global_idx = 0

for f in parquet_files:
    # Load the whole file to preserve all sensor/action data
    df = pd.read_parquet(f)
    
    # Calculate how many frames are in this specific file
    num_rows = len(df)
    
    # Create a perfectly sequential list of numbers starting from our current position
    # This turns [12481, 12772...] into [12481, 12482...]
    new_indices = list(range(current_global_idx, current_global_idx + num_rows))
    
    # Overwrite only the 'index' column
    df['index'] = new_indices
    
    # Save the file back to its original location
    df.to_parquet(f, index=False)
    
    print(f"✅ Fixed {f.name}: Re-indexed to {new_indices[0]} through {new_indices[-1]}")
    
    # Move the pointer forward for the next file in the sequence
    current_global_idx += num_rows

print(f"\nAlignment Complete!")
print(f"The new 'total_frames' for your info.json should be: {current_global_idx}")