import pandas as pd
from pathlib import Path

# Path to your first data parquet
path = Path("~/.cache/huggingface/lerobot/waly/multiple_places/data/chunk-000/file-000.parquet").expanduser()

# Load the entire file structure
df = pd.read_parquet(path)

# Print the shape (Rows, Columns) to see the scale
print(f"File Shape: {df.shape} (Rows, Columns)")
print("-" * 30)

# Display the first 5 rows with all columns
# Use .T (Transpose) if you have many columns and want to see them as a list
print("--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN LIST & DATA TYPES ---")
print(df.dtypes)