# -*- coding: utf-8 -*-
import pandas as pd

def main():
    dfs = []
    for tag in ["wild", "bei", "tie"]:
        p = f"assets/reprocessed_hcsu_{tag}.csv"
        df = pd.read_csv(p)
        df["source"] = f"hcsu_{tag}"
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv("assets/reprocessed_hcsu_all.csv", index=False)
    print(f"Unified CSV saved: assets/reprocessed_hcsu_all.csv, total rows: {len(all_df)}")
    print("Source counts:")
    print(all_df["source"].value_counts())

if __name__ == "__main__":
    main()
