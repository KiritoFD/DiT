import csv
from pathlib import Path

csv_p = Path("exp-std/csv/train_top10_noaug.csv")
rows = list(csv.DictReader(open(csv_p, encoding="utf-8-sig")))
print("train_top10_noaug:", len(rows), "rows")
print("Sample row:", rows[0])
