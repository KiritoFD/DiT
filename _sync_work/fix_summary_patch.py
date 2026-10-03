"""修 P0-1: in_mem_eval.py 的 summary 写入被自己改名的孤儿文件吞掉。

## 实测病灶 (2026-10-03)
runs_AB/ 下**没有** eval_stdskel_summary.csv, 只有 3 个 `.bak_oldcols*`,
每个恰好含一次 eval 的表头 + 1 行数据 —— 即**每次 eval 的数据都写进了被改名的旧 inode**。

## 根因 (两段, 缺一不可)
1. `f_sum = open(sum_path, "a")` 在表头检查**之前**执行 -> append 模式**创建空文件**
   -> 后面 `os.path.exists(sum_path)` 恒真、`next(csv.reader)` 恒读到 0 列
   -> 每次 eval 都误判"表头不匹配"(实测日志 "0 vs 22 列")。
2. 判定不匹配后 `os.rename(sum_path, _bak)`, 但 **f_sum 句柄仍指向旧 inode** ->
   之后 `w_sum.writerow(...)` 全部写进改名后的备份文件 -> 磁盘上主文件永不存在。

## 修法
把 `_SUM_HDR` 定义 + 表头检查整段**挪到 open 之前**; 检查通过才 open;
并加一行累积自检 (行数 + 已有 (step,set) 对数), 让"曲线被重置"这类问题下次立刻暴露。

事务式 + 幂等 + 语法自检。
"""
import io
import os
import re
import shutil
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
F = "src/eval/in_mem_eval.py"

SUM_HDR = ('["exp", "step", "set", "n", "ssim_mean", "ssim_p10", "ssim_q1",\n'
           '                "ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean",\n'
           '                "ink_ssim_mean", "ink_iou_mean", "skel_iou_mean",\n'
           '                "frag_ratio", "hole_pred", "hole_gt",\n'
           '                "nn_ssim", "nn_mean", "tgt_spec", "cal_enrich"]')

NEW = '''    sum_path = os.path.join(results_dir, "eval_stdskel_summary.csv")
    raw_path = os.path.join(results_dir, "eval_stdskel_batch.csv")
    # ★★ 2026-10-03 修 (P0-1): 表头检查必须放在 open(..., "a") **之前**。
    #   旧顺序是 [open("a") -> exists() -> 读表头], 而 "a" 模式会**创建空文件**,
    #   于是 exists() 恒真、表头恒读到 0 列 -> 每次 eval 都误判"表头不匹配"。
    #   更致命的是 rename 时 f_sum 句柄仍指向旧 inode, writer 继续往**改名后的
    #   备份文件**里写 -> 磁盘上主 summary 永远不存在 (实测 runs_AB 只有 3 个
    #   .bak_oldcols 而无主文件, 每个 .bak 恰好含一次 eval 的数据)。
    _SUM_HDR = __SUM_HDR__
    if os.path.exists(sum_path):
        try:
            with open(sum_path, encoding="utf-8") as _f:
                _got = next(csv.reader(_f), [])
        except StopIteration:
            _got = []
        if _got != _SUM_HDR:
            _bak = sum_path + ".bak_oldcols"
            if os.path.exists(_bak):
                _bak = sum_path + f".bak_oldcols.{int(os.path.getmtime(sum_path))}"
            os.rename(sum_path, _bak)
            print(f"[in-mem-eval] ⚠ summary 表头不匹配({len(_got)} vs {len(_SUM_HDR)} 列) "
                  f"-> 旧文件备份为 {os.path.basename(_bak)}, 重开新表")
    # 检查完再读已完成项 (此时文件要么不存在, 要么表头已验证)
    done = set()
    if os.path.exists(sum_path):
        for r in csv.DictReader(open(sum_path, encoding="utf-8")):
            done.add((int(r["step"]), r["set"]))
    new_sum = not os.path.exists(sum_path)
    new_raw = not os.path.exists(raw_path)
    # ★ 句柄只在表头确认之后才打开; 之后不会再 rename 本文件 -> 不会再产生孤儿写入。
    f_sum = open(sum_path, "a", newline="", encoding="utf-8")
    f_raw = open(raw_path, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    w_raw = csv.writer(f_raw)
    if new_sum:
        w_sum.writerow(_SUM_HDR)
        f_sum.flush()
    # ★ 累积自检: 每次 eval 打印"主文件行数 + 已完成 (step,set) 对数"。
    #   若曲线再被重置, 这行会立刻显示行数归零/不增长 -> 一眼可见, 不必事后考古。
    try:
        _nlines = sum(1 for _ in open(sum_path, encoding="utf-8")) if os.path.exists(sum_path) else 0
    except Exception:                                            # noqa: BLE001
        _nlines = -1
    print(f"[eval] summary {sum_path}: {_nlines} 行, 已完成 (step,set) {len(done)} 对, "
          f"表头 {len(_SUM_HDR)} 列{' (本次新建)' if new_sum else ''}", flush=True)
'''

ANCHOR_START = '    sum_path = os.path.join(results_dir, "eval_stdskel_summary.csv")'
ANCHOR_END = "    if new_sum:\n        w_sum.writerow(_SUM_HDR)"


def main():
    src = io.open(F, encoding="utf-8").read()
    if "[eval] summary " in src and "[P0-1]" in src:
        print(f"[skip] {F} 已修 (幂等)")
        return 0
    i = src.find(ANCHOR_START)
    j = src.find(ANCHOR_END)
    if i < 0 or j < 0 or j < i:
        print(f"[FATAL] 锚点缺失 (start={i}, end={j}) -> 不做任何修改")
        return 2
    j += len(ANCHOR_END)
    old = src[i:j]
    print("=" * 78)
    print(f"将替换 {F} 的第 {src[:i].count(chr(10)) + 1} 行起, 共 {old.count(chr(10)) + 1} 行")
    print("  旧片段首行:", old.splitlines()[0])
    print("  旧片段末行:", old.splitlines()[-1])
    print("=" * 78)

    ts = time.strftime("%Y%m%d-%H%M%S")
    bdir = f"_sync_work/_AB_patch/backup_summary_{ts}"
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(F, os.path.join(bdir, "in_mem_eval.py"))
    print(f"[backup] -> {bdir}/in_mem_eval.py")

    new = src[:i] + NEW.replace("__SUM_HDR__", SUM_HDR) + src[j:]
    io.open(F, "w", encoding="utf-8", newline="").write(new)
    print(f"[write] {F}")
    import py_compile
    try:
        py_compile.compile(F, doraise=True, cfile="/tmp/_chk4.pyc")
        print("  ✓ 语法通过")
    except py_compile.PyCompileError as e:
        print(f"  ✗ 语法错误\n{e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
