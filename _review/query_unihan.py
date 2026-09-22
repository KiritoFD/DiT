"""查 Unihan 里「升」「陞」的变体关系。"""
import re

P = "/tmp/unihan/Unihan_Variants.txt"
targets = {0x5347: "升", 0x965E: "陞", 0x6607: "昇"}
lines = open(P, encoding="utf-8", errors="ignore").read().splitlines()

print("  === Unihan_Variants.txt 里的变体字段类型 ===")
kinds = {}
for ln in lines:
    if ln.startswith("#") or not ln.strip():
        continue
    parts = ln.split("\t")
    if len(parts) >= 3:
        kinds[parts[1]] = kinds.get(parts[1], 0) + 1
for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
    print(f"    {k:<28} {v}")

print("\n  === 直接查 升/陞/昇 ===")
for cp, ch in targets.items():
    print(f"\n  {ch} U+{cp:04X}:")
    found = False
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) >= 3 and parts[0].lower() == f"u+{cp:04x}":
            found = True
            # 值形如 "U+965E<kMatthews" 或多个
            refs = re.findall(r"U\+([0-9A-Fa-f]{4,5})", parts[2])
            chars = "".join(chr(int(r, 16)) for r in refs)
            print(f"    {parts[1]:<24} -> {chars}  ({parts[2][:80]})")
    if not found:
        print("    （无记录）")

print("\n  === 反向：谁的变体里有 升 或 陞 ===")
for ln in lines:
    if ln.startswith("#") or not ln.strip():
        continue
    parts = ln.split("\t")
    if len(parts) >= 3 and ("5347" in parts[2].upper() or "965E" in parts[2].upper()):
        cp = parts[0]
        try:
            ch = chr(int(cp[2:], 16))
        except ValueError:
            ch = "?"
        print(f"    {ch} {cp}  {parts[1]}  -> {parts[2][:70]}")
