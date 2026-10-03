"""P0-1 单测: 复现旧顺序的 bug + 证明新顺序不会再犯。

不依赖仓库任何模块, 纯 stdlib, 在临时目录里跑两轮 "eval"。
"""
import csv
import os
import shutil
import tempfile
import time

HDR = ["exp", "step", "set", "n"]
ROW = lambda step: ["runs_AB", str(step), "eval200fix", "187"]


def old_order(d, step):
    """旧顺序: 先 open("a") 再查表头 (实测线上版本)。"""
    sp = os.path.join(d, "summary.csv")
    done = set()
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding="utf-8")):
            done.add(r["step"])
    new_sum = not os.path.exists(sp)
    f = open(sp, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if os.path.exists(sp):                       # ← 此时恒真 (被上面的 "a" 创建了)
        got = next(csv.reader(open(sp, encoding="utf-8")), [])
        if got != HDR:
            bak = sp + ".bak_oldcols"
            if os.path.exists(bak):
                bak = sp + f".bak_oldcols.{int(os.path.getmtime(sp))}"
            os.rename(sp, bak)                   # ← 句柄 f 仍指向旧 inode
            done.clear()
    new_sum = not os.path.exists(sp)
    if new_sum:
        w.writerow(HDR)
    w.writerow(ROW(step))
    f.flush()
    f.close()                                    # 数据落到"改名后的备份"里
    return sp


def new_order(d, step):
    """新顺序: 先查表头, 通过后才 open。"""
    sp = os.path.join(d, "summary.csv")
    if os.path.exists(sp):
        try:
            got = next(csv.reader(open(sp, encoding="utf-8")), [])
        except StopIteration:
            got = []
        if got != HDR:
            bak = sp + ".bak_oldcols"
            if os.path.exists(bak):
                bak = sp + f".bak_oldcols.{int(os.path.getmtime(sp))}"
            os.rename(sp, bak)
    done = set()
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding="utf-8")):
            done.add(r["step"])
    new_sum = not os.path.exists(sp)
    f = open(sp, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_sum:
        w.writerow(HDR)
    w.writerow(ROW(step))
    f.flush()
    f.close()
    return sp


def run(label, fn):
    d = tempfile.mkdtemp()
    try:
        fn(d, 5000)
        time.sleep(1.1)                          # 让 mtime 不同, 避免 .bak 名冲突
        fn(d, 10000)
        files = sorted(os.listdir(d))
        main = os.path.join(d, "summary.csv")
        n_main = 0
        if os.path.exists(main):
            n_main = sum(1 for _ in open(main, encoding="utf-8"))
        baks = [f for f in files if ".bak_oldcols" in f]
        print(f"\n=== {label} ===")
        print(f"  目录: {files}")
        print(f"  主文件存在: {os.path.exists(main)}   主文件行数: {n_main}")
        print(f"  备份文件数: {len(baks)}   {baks}")
        for b in baks:
            p = os.path.join(d, b)
            print(f"    {b}: {sum(1 for _ in open(p, encoding='utf-8'))} 行")
        return os.path.exists(main), n_main, len(baks)
    finally:
        shutil.rmtree(d, ignore_errors=True)


print("=" * 74)
print("P0-1 单测: 旧顺序 vs 新顺序 (两轮 eval)")
print("=" * 74)
ok_old, n_old, b_old = run("旧顺序 (线上实测版本)", old_order)
ok_new, n_new, b_new = run("新顺序 (本次修复)", new_order)

print("\n" + "=" * 74)
print("判定:")
print(f"  旧顺序: 主文件存在={ok_old} 行数={n_old} 备份={b_old}  "
      f"-> {'✗ 复现了 bug (主文件消失/曲线丢进备份)' if (not ok_old or b_old > 0) else '未复现'}")
print(f"  新顺序: 主文件存在={ok_new} 行数={n_new} 备份={b_new}  "
      f"-> {'✓ 修复 (主文件累积, 无孤儿备份)' if (ok_new and n_new == 3 and b_new == 0) else '✗ 仍有问题'}")
print("  期望: 新顺序 = 1 行表头 + 2 行数据 = 3 行, 且 0 个备份")
