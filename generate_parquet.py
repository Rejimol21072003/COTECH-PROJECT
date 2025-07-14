import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed
np.random.seed(42)
n = 500  # number of sample rows

# Generate fake pickup/dropoff times
pickup_times = [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 10000)) for _ in range(n)]
durations = [timedelta(minutes=np.random.randint(5, 60)) for _ in range(n)]
dropoff_times = [pickup_times[i] + durations[i] for i in range(n)]

# Create DataFrame
df = pd.DataFrame({
    "pickup_datetime": pickup_times,
    "dropoff_datetime": dropoff_times,
    "passenger_count": np.random.randint(1, 5, size=n),
    "trip_distance": np.round(np.random.uniform(1.0, 15.0, size=n), 2),
    "fare_amount": np.round(np.random.uniform(5.0, 50.0, size=n), 2),
    "pickup_latitude": np.random.uniform(40.70, 40.85, size=n),
    "pickup_longitude": np.random.uniform(-74.02, -73.93, size=n),
    "dropoff_latitude": np.random.uniform(40.70, 40.85, size=n),
    "dropoff_longitude": np.random.uniform(-74.02, -73.93, size=n),
})

# Make the directory if not exists
os.makedirs("data", exist_ok=True)

# Save as Parquet
df.to_parquet("data/sample_trips.parquet")

print("✅ sample_trips.parquet file created in 'data/' folder.")
