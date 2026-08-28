"""清洗: 只保留 GB2312 一级+二级(简体常用/次常用)字符的训练/评估集."""
import csv

def gb2312_chars():
    chars = set()
    for q in range(0xB0, 0xF8):      # 16-87 区 (一级+二级)
        for p in range(0xA1, 0xFF):
            try:
                chars.add(bytes([q, p]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return chars

KEEP = gb2312_chars()
print(f"GB2312 一级+二级: {len(KEEP)} 字")

def filter_csv(src, dst):
    n_in = n_out = 0
    dropped = {"rows": 0, "chars": set()}
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows_out = []
        for r in reader:
            n_in += 1
            if r["character"] in KEEP:
                rows_out.append(r)
                n_out += 1
            else:
                dropped["rows"] += 1
                dropped["chars"].add(r["character"])
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"{src} -> {dst}: {n_in} -> {n_out} rows, dropped {dropped['rows']} rows / {len(dropped['chars'])} chars")
    return rows_out

filter_csv("5script/train_3top30_nobeike.csv", "5script/train_3top30_common.csv")
filter_csv("5script/eval500_3top30.csv", "5script/eval500_3top30_common.csv")