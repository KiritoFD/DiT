import csv, collections, os, sys
import pandas as pd

p = "assets/train_fame-kxl-tj-px60_sym.csv"
df = pd.read_csv(p)
print("rows", len(df))
print("cols", list(df.columns))
print()
print("img subdirs:", collections.Counter(
    str(x).split("/")[1] for x in df["image_path"]).most_common(6))
print("std subdirs:", collections.Counter(
    str(x).split("/")[1] for x in df["std_path"]).most_common(6))
if "aug" in df.columns:
    print("aug values:", collections.Counter(df["aug"].fillna("")).most_common(6))
print()
print("calligraphers:", df["calligrapher"].nunique())
print("scripts:", collections.Counter(df["script"]).most_common())
print()
# check existence of a few files
for pre in ["image_path", "std_path"]:
    miss = 0
    for v in df[pre].head(300):
        if not os.path.exists(str(v)):
            miss += 1
    print(f"{pre}: {miss}/300 missing in first 300")
print()
print("--- sample rows ---")
print(df.head(3).to_string())
