import pandas as pd
from pathlib import Path

input_file = Path(r"C:\Users\user\.ai_network_monitor\training_dataset.csv")
output_file = Path(r"C:\Users\user\.ai_network_monitor\isolation_forest_train_normal.csv")

df = pd.read_csv(input_file)

print("Original rows:", len(df))
print("Label counts:")
print(df["label"].value_counts())

normal_df = df[df["label"] == 0].copy()

print("Normal rows:", len(normal_df))

normal_df.to_csv(output_file, index=False)

print("Saved:", output_file)