# -*- coding: utf-8 -*-
"""扫描所有训练日志，提取指标行的「格式模式」（数字归一化为 <N>），用于写通用解析器。

输出：按频次排序的去重模式列表，帮助一次性看清历史上出现过的所有日志格式。
"""
import os, sys, re, glob, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_GLOBS = ["/tmp/*.log", "5script/results/*/*/log.txt",
             "5script/results/*/log.txt", "logs/*.log"]

KEY = re.compile(r"(ssim|lpips|mse|fid|psnr|eval step|gpu-eval|early-stop|best)", re.I)


def norm(line: str) -> str:
    # 去掉 ANSI 颜色
    s = re.sub(r"\x1b\[[0-9;]*m", "", line)
    # 时间戳归一
    s = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "<TS>", s)
    # 浮点/整数归一
    s = re.sub(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", "<N>", s)
    # 路径归一
    s = re.sub(r"[\w./\\-]+\.(?:pt|npz|json|csv|png|npy)", "<PATH>", s)
    s = re.sub(r"step_?\d+", "step_<N>", s)
    return s.strip()


def main():
    files = []
    for g in LOG_GLOBS:
        files.extend(sorted(glob.glob(g)))
    files = sorted(set(files))
    print(f"# log files found: {len(files)}")

    counter = collections.Counter()
    example = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not KEY.search(line):
                        continue
                    if len(line) > 400:
                        continue
                    p = norm(line)
                    counter[p] += 1
                    example.setdefault(p, (fp, line.strip()[:220]))
        except Exception as e:
            print(f"# skip {fp}: {e}")

    print(f"\n# distinct patterns: {len(counter)}\n")
    for pat, cnt in counter.most_common(60):
        fp, raw = example[pat]
        print(f"[x{cnt}] {os.path.basename(fp)}")
        print(f"    PAT : {pat}")
        print(f"    RAW : {raw}")
        print()


if __name__ == "__main__":
    main()
