import csv
from pathlib import Path

# The 30 selected indices from eval_top5_fixed.csv
SELECTED_INDICES = [
    # 欧阳询 (3 Seen, 3 Unseen)
    (2,  "seen"),    # 太 (楷)
    (3,  "seen"),    # 切 (楷)
    (11, "seen"),    # 得 (行)
    (1,  "unseen"),  # 率 (楷)
    (4,  "unseen"),  # 快 (楷)
    (17, "unseen"),  # 枢 (行)

    # 王羲之 (3 Seen, 3 Unseen)
    (18, "seen"),    # 旨 (楷)
    (25, "seen"),    # 卅 (楷)
    (29, "seen"),    # 羲 (行)
    (19, "unseen"),  # 尚 (楷)
    (26, "unseen"),  # 好 (行)
    (28, "unseen"),  # 夜 (行)

    # 苏轼 (3 Seen, 3 Unseen)
    (39, "seen"),    # 入 (楷)
    (44, "seen"),    # 危 (行)
    (47, "seen"),    # 渚 (行)
    (35, "unseen"),  # 靑 (楷)
    (40, "unseen"),  # 土 (楷)
    (48, "unseen"),  # 冒 (行)

    # 赵孟頫 (3 Seen, 3 Unseen)
    (52, "seen"),    # 對 (楷)
    (66, "seen"),    # 寂 (行)
    (75, "seen"),    # 字 (隶)
    (55, "unseen"),  # 丞 (楷)
    (62, "unseen"),  # 府 (行)
    (74, "unseen"),  # 伐 (隶)

    # 颜真卿 (3 Seen, 3 Unseen)
    (78, "seen"),    # 其 (楷)
    (81, "seen"),    # 妃 (楷)
    (90, "seen"),    # 深 (行)
    (80, "unseen"),  # 誦 (楷)
    (87, "unseen"),  # 風 (行)
    (89, "unseen"),  # 空 (行)
]

eval_top5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))

top5_30_rows = []
for orig_idx, status in SELECTED_INDICES:
    r = dict(eval_top5[orig_idx])
    r["is_seen"] = status
    r["orig_eval_idx"] = orig_idx
    top5_30_rows.append(r)

out_path = Path("exp-std/csv/eval_top5_30.csv")
fieldnames = list(top5_30_rows[0].keys())

with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(top5_30_rows)

print(f"Successfully generated {out_path} with {len(top5_30_rows)} samples:")
for i, r in enumerate(top5_30_rows):
    st = r["is_seen"].upper()
    print(f"#{i+1:02d} (orig {r['orig_eval_idx']:02d}) [{st:6s}] {r['calligrapher']} · {r['script']}书 【{r['character']}】")
