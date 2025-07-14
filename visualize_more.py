import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("enhanced_trips.parquet")

# Plot 1: Distance vs Time
plt.figure(figsize=(6,4))
plt.scatter(df['trip_distance'], df['trip_time_minutes'], color='blue')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Time (minutes)')
plt.title('Distance vs Time')
plt.grid(True)
plt.show()

# Plot 2: Passengers vs Fare
plt.figure(figsize=(6,4))
plt.scatter(df['passenger_count'], df['fare_amount'], color='green')
plt.xlabel('Passenger Count')
plt.ylabel('Fare Amount ($)')
plt.title('Passengers vs Fare')
plt.grid(True)
plt.show()
