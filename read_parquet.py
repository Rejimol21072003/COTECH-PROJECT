import pandas as pd

# Load the Parquet file
df = pd.read_parquet("sample_trips.parquet")

# Display basic info
print("✅ Parquet file loaded successfully!")
print("\n📋 First 5 rows:\n", df.head())
print("\n📊 Summary:\n", df.describe())
