import pandas as pd

df = pd.read_csv("/root/Workspace/xy/DiT/assets/train_50k_v2.csv")
print("Total rows:", len(df))
print(df[["image_path", "std_path"]].head(10))
