import pandas as pd
df = pd.read_csv("data/metadata_augmented.csv")
print(df.groupby(["roof_type", "connection_type"]).size().unstack(fill_value=0))
print("\nAngle distribution:\n", df["angle"].describe())
print("\nPanel count distribution:\n", df["num_panels"].describe())