import pandas as pd

# Load the original CSV with windowed features
input_csv = "features_windowed_yawn_distinct.csv"
output_csv = "features_windowed_yawn_nonzero.csv"

# Load the data
df = pd.read_csv(input_csv)

# Filter rows where YawnCount is not zero
filtered_df = df[df["YawnCount"] > 0]

# Save the filtered data to a new CSV
filtered_df.to_csv(output_csv, index=False)

output_csv
