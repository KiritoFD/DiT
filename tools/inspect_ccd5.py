# -*- coding: utf-8 -*-
"""read the csv inside chinese-calligraphy-dataset-221030 zip + full char dir stats."""
import csv
import collections
import io
import zipfile

Z = r"E:\chinese-calligraphy-dataset-221030 (1).zip"
z = zipfile.ZipFile(Z)


def dec(n):
    if "__MACOSX" in n:
        return None
    try:
        return n.encode("cp437").decode("utf-8")
    except Exception:
        return n


csv_name = None
for zi in z.infolist():
    d = dec(zi.filename)
    if d and d.lower().endswith(".csv"):
        csv_name = zi.filename
        break
with z.open(csv_name) as f:
    text = f.read().decode("utf-8", errors="replace")
lines = text.splitlines()
print("csv lines:", len(lines))
print("header:", lines[0])
print("rows:", lines[1:4])

# char dirs with counts
names = [dec(n) for n in z.namelist()]
files = [n for n in names if n and n.lower().endswith(".jpg")]
char_cnt = collections.Counter()
for n in files:
    parts = n.split("/")
    if len(parts) >= 4:
        char_cnt[parts[2]] += 1
print(f"unique chars: {len(char_cnt)}, total jpgs: {sum(char_cnt.values())}")
top = char_cnt.most_common(10)
print("top chars:", top)
bottom = sorted(char_cnt.items(), key=lambda x: x[1])[:10]
print("rarest:", bottom)
