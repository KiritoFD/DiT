import pandas as pd
import os

df = pd.read_csv("/root/Workspace/xy/DiT/exp-std/csv/eval200_fixed.csv")
print(f"Total samples: {len(df)}")

# Find interesting samples across calligraphers and scripts
grouped = df.groupby(["calligrapher", "script"])
selected = []
for (callig, script), g in grouped:
    # Pick the first one from each group
    idx = g.index[0]
    row = g.iloc[0]
    selected.append({
        "idx": idx,
        "callig": callig,
        "script": script,
        "char": row["character"],
        "img_id": row["img_id"],
        "image_path": row["image_path"]
    })

res_df = pd.DataFrame(selected)
print(res_df.to_string())

# Pick 8 clean, diverse indices
sample_indices = res_df["idx"].tolist()[:10]
print("\nSelected indices:", sample_indices)
