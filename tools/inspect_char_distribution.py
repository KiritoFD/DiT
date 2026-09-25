import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'assets/train_50k_v2.csv'
df = pd.read_csv(csv_path)

print("="*60)
print(f"Dataset: {csv_path}")
print(f"Total samples: {len(df)}")
print(f"Distinct characters: {df['character'].nunique()}")
print(f"Distinct calligraphers: {df['calligrapher'].nunique()}")
print(f"Scripts: {df['script'].value_counts().to_dict()}")

char_counts = df['character'].value_counts()
c1 = (char_counts == 1).sum()
c2 = (char_counts == 2).sum()
c_le2 = (char_counts <= 2).sum()
c_le5 = (char_counts <= 5).sum()
total_chars = len(char_counts)

print("\n--- [Character Frequency] ---")
print(f"Char count == 1: {c1} ({c1/total_chars*100:.2f}%)")
print(f"Char count == 2: {c2} ({c2/total_chars*100:.2f}%)")
print(f"Char count <= 2: {c_le2} ({c_le2/total_chars*100:.2f}%)")
print(f"Char count <= 5: {c_le5} ({c_le5/total_chars*100:.2f}%)")
print(f"Median char count: {char_counts.median()}, Mean: {char_counts.mean():.2f}, Max: {char_counts.max()}")

df['char_script'] = df['character'].astype(str) + '_' + df['script'].astype(str)
cs_counts = df['char_script'].value_counts()
cs1 = (cs_counts == 1).sum()
cs2 = (cs_counts == 2).sum()
cs_le2 = (cs_counts <= 2).sum()
cs_le5 = (cs_counts <= 5).sum()
total_cs = len(cs_counts)

print("\n--- [Character * Script Frequency] ---")
print(f"Distinct (char, script): {total_cs}")
print(f"(char, script) count == 1: {cs1} ({cs1/total_cs*100:.2f}%)")
print(f"(char, script) count == 2: {cs2} ({cs2/total_cs*100:.2f}%)")
print(f"(char, script) count <= 2: {cs_le2} ({cs_le2/total_cs*100:.2f}%)")
print(f"(char, script) count <= 5: {cs_le5} ({cs_le5/total_cs*100:.2f}%)")
print(f"Median count: {cs_counts.median()}, Mean: {cs_counts.mean():.2f}, Max: {cs_counts.max()}")

print("\n--- [Std Skeleton Sharing] ---")
if 'std_path' in df.columns:
    std_counts = df['std_path'].value_counts()
    print(f"Distinct std_paths: {len(std_counts)}")
    print(f"Samples per std_path: Mean={std_counts.mean():.2f}, Median={std_counts.median()}, Max={std_counts.max()}")
    print(f"Std paths shared by > 1 sample: {(std_counts > 1).sum()} ({(std_counts > 1).sum()/len(std_counts)*100:.2f}%)")
    print(f"Std paths used by only 1 sample: {(std_counts == 1).sum()} ({(std_counts == 1).sum()/len(std_counts)*100:.2f}%)")

print("="*60)
