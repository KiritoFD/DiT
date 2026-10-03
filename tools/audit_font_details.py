import csv
from collections import Counter

csv_path = "assets/train_top10_style23.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

syn_rows = [r for r in rows if "font" in r.get("source", "")]
fonts = Counter(r.get("src_image_path", "") for r in syn_rows)
print(f"Total synthetic rows: {len(syn_rows)}")
print("Synthetic fonts breakdown:")
for font, count in fonts.most_common():
    print(f"  {font:35s}: {count:5d} ({count/len(syn_rows)*100:4.1f}%)")

# Also check slot to font mapping
slot_font = Counter((r["slot_name"], r.get("src_image_path", "")) for r in syn_rows)
print("\nSlot to Font breakdown:")
for (slot, font), count in slot_font.most_common(20):
    print(f"  {slot:16s} -> {font:25s}: {count:5d}")
