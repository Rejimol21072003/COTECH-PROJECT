import pandas as pd
import matplotlib.pyplot as plt

# Load the parquet file
df = pd.read_parquet("sample_trips.parquet")

# Plot fare amount vs trip distance
plt.figure(figsize=(8, 5))
plt.scatter(df["trip_distance"], df["fare_amount"], color="skyblue", edgecolor="black")
plt.title("Trip Distance vs Fare Amount")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount ($)")
plt.grid(True)
plt.tight_layout()
plt.show()
    