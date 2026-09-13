# -*- coding: utf-8 -*-
"""
_audit_remote_files.py — 远程文件全量清点，按规则给出「保留/归档/删除/写入文档」建议。

分类规则
--------
DELETE（肯定没用）:
  - 过期缓存 __pycache__ / *.pyc
  - 空文件或 0 字节产物
  - 旧实验结果的中间 PNG（eval_samples 里的海量图片）
  - 名字里带 _old/_bak/_tmp/backup 且已被新版取代
  - 超过 30 天未动且属于一次性调试产物（_check_*/_probe_* 等）

ARCHIVE（整理归档）:
  - 早期系列（s2–s14）的配置与结果目录
  - 已被取代但仍可能要对照的脚本版本

KEEP（活跃）:
  - src/ 下的源码
  - src/train/configs 下当前系列的配置
  - tools/ 下当前在用的工具
  - docs/ 文档

输出: assets/file_audit.csv （供人工复核后执行）
"""
import os, sys, csv, time, fnmatch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.getcwd()
NOW = time.time()
DAY = 86400

# 明确不进 git / 无需审计的大目录（数据产物）
SKIP_DIRS = {
    "data/imgs/final_imgs_256", "final_images", "final_canny", "data/skel/final_skeleton",
    "data/skel/final_skeleton_d3", "final_latents", "data/latents/final_latents_fame",
    "data/latents/final_latents_f4", "data/latents/final_latents_mid_clean",
    "data/skel/final_skel_latents_fame", "data/skel/final_skel_latents_fame_1px",
    "data/skel/final_skel_latents_mid_clean", "data/skel/final_skel_latents_train_1px",
    "data/skel/final_skel_latents_eval_1px", "data/skel/final_skel3_fame", "data/skel/final_skel1_fame",
    "data/skel/final_skel3_mid_clean", "data/skel/final_skel1", "data/skel/final_skel3", "data/skel/final_skel1",
    "pretrained_models", ".git", "__pycache__", "_gt_cache", "dataset",
    "MCCD", "logs", "wandb", ".codebuddy",
}

# 早期系列（归档候选）
OLD_SERIES = re_old = None
import re
OLD_RE = re.compile(r"^(s\d+|v3[a-z]?|exp_s|compositional|px_s|skel_decoder|"
                    r"structnet|struct_decoder|probes|latent_struct)")


def size_bytes(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def size_mb(p):
    b = size_bytes(p)
    return round(b / 1024 / 1024, 3) if b >= 0 else -1


def dir_size_mb(p):
    tot = 0
    cnt = 0
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
                cnt += 1
            except OSError:
                pass
    return round(tot / 1024 / 1024, 2), cnt


def classify(rel, isdir, size, mtime, ext, nbytes=1):
    age = (NOW - mtime) / DAY
    base = os.path.basename(rel)

    # ---- DELETE ----
    if ext in (".pyc",) or "__pycache__" in rel:
        return "DELETE", "字节码缓存"
    # 注意：必须用原始字节数判断空文件。早期版本用 round(MB,3) 会把
    # 几百字节的小脚本也舍成 0.0，导致误判为"空文件"而建议删除源码。
    if not isdir and nbytes == 0:
        return "DELETE", "空文件(0字节)"
    if fnmatch.fnmatch(base, "*_old.*") or fnmatch.fnmatch(base, "*.bak") \
            or fnmatch.fnmatch(base, "*_bak.*") or fnmatch.fnmatch(base, "*.orig"):
        return "DELETE", "旧版备份"
    # 一次性调试脚本：以 _ 开头且超过 14 天
    if base.startswith("_") and ext == ".py" and age > 14:
        return "DELETE", f"一次性调试脚本({age:.0f}天未动)"
    if isdir and "eval_samples" in rel:
        return "DELETE", "评测中间图(可重生成)"
    if isdir and "_cleanup_" in base:
        return "DELETE", "清理暂存目录"

    # ---- ARCHIVE ----
    if isdir and rel.startswith("assets/results/"):
        series = rel.split("/")[2] if len(rel.split("/")) > 2 else ""
        if OLD_RE.match(series) and series not in (
                "s20_midcommon_s_flow_v2", "s21_fame_flow_v2",
                "s22_char_embed_only", "s23_glyph_cond",
                "ctrl_fame_v2", "ctrl_fame_1pix_v1"):
            return "ARCHIVE", f"早期系列 {series}"
    if isdir and rel.startswith("_archive"):
        return "ARCHIVE", "已归档区"

    # ---- KEEP ----
    if rel.startswith("src/"):
        return "KEEP", "源码"
    if rel.startswith("docs/"):
        return "KEEP", "文档"
    if rel.startswith("tools/") and ext == ".py":
        return "KEEP", "工具脚本"
    if rel.startswith("configs/") or rel.startswith("src/train/configs/"):
        return "KEEP", "配置"
    if rel.startswith("assets/") and ext in (".csv", ".json"):
        return "KEEP", "数据/结果表"

    return "REVIEW", "需人工判断"


def main():
    rows = []
    for entry in sorted(os.listdir(ROOT)):
        if entry in SKIP_DIRS or entry.startswith("."):
            continue
        p = os.path.join(ROOT, entry)
        rel = entry
        if os.path.isdir(p):
            sz, cnt = dir_size_mb(p)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                mt = NOW
            act, why = classify(rel, True, sz, mt, "", nbytes=cnt)
            rows.append({"path": rel + "/", "type": "dir", "size_mb": sz,
                         "n_files": cnt,
                         "age_days": round((NOW - mt) / DAY, 1),
                         "action": act, "reason": why})
        else:
            ext = os.path.splitext(entry)[1].lower()
            try:
                mt = os.path.getmtime(p)
            except OSError:
                mt = NOW
            nb = size_bytes(p)
            sz = size_mb(p)
            act, why = classify(rel, False, sz, mt, ext, nbytes=nb)
            rows.append({"path": rel, "type": "file", "size_mb": sz,
                         "n_files": 1,
                         "age_days": round((NOW - mt) / DAY, 1),
                         "action": act, "reason": why})

    # 细分：把 KEEP 的大目录展开一层（tools / src / assets/results）
    extra = []
    for sub in ("tools", "src/eval", "src/model", "src/train", "src/utils", "src/loss"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isdir(p) or not fn.endswith(".py"):
                continue
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            extra.append({"path": f"{sub}/{fn}", "type": "file",
                          "size_mb": size_mb(p), "n_files": 1,
                          "age_days": round((NOW - mt) / DAY, 1),
                          "action": "KEEP", "reason": f"{sub} 源码"})
    rows.extend(extra)

    os.makedirs("assets", exist_ok=True)
    cols = ["path", "type", "size_mb", "n_files", "age_days", "action", "reason"]
    with open("assets/file_audit.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["action"], -x["size_mb"])):
            w.writerow(r)

    # ── 专项核查：标准字形库的真实位置 ────────────────────────────────────
    # v1 代码 (src/utils/glyph_latent.py) 找的是 src/utils/std_glyph_latent，
    # 但根目录另有一个 std_glyph_latent/。位置不对会导致命中率 0%。
    print("\n" + "=" * 70)
    print("# 专项：标准字形库位置核查（关系到 w_glyph_cond 是否可用）")
    print("=" * 70)
    for cand in ("src/utils/std_glyph_latent", "std_glyph_latent",
                 "src/utils/std_glyph_latent_v2", "std_glyph_latent_v2"):
        if os.path.isdir(cand):
            fonts = {}
            for f in sorted(os.listdir(cand)):
                d = os.path.join(cand, f)
                if os.path.isdir(d):
                    fonts[f] = len([x for x in os.listdir(d) if x.endswith(".npy")])
            print(f"  EXISTS  {cand:<34} 字体={fonts}")
        else:
            print(f"  MISSING {cand}")
    for tgz in ("std_glyph_latent.tgz", "std_glyph_latent.tar.gz",
                "std_glyph_latent_v2.tgz"):
        if os.path.isfile(tgz):
            print(f"  TGZ     {tgz:<34} {size_mb(tgz)}MB (解压可得)")

    import collections
    cnt = collections.Counter(r["action"] for r in rows)
    tot = collections.defaultdict(float)
    for r in rows:
        tot[r["action"]] += r["size_mb"]
    print(f"# wrote assets/file_audit.csv ({len(rows)} entries)")
    for k in ("DELETE", "ARCHIVE", "KEEP", "REVIEW"):
        print(f"  {k:<8} n={cnt[k]:<4} {tot[k]:>10.1f} MB")

    print("\n# DELETE 明细（按大小）:")
    for r in sorted([x for x in rows if x["action"] == "DELETE"],
                    key=lambda x: -x["size_mb"])[:25]:
        print(f"  {r['size_mb']:>9.1f}MB  {r['path']:<45} {r['reason']}")

    print("\n# ARCHIVE 明细:")
    for r in sorted([x for x in rows if x["action"] == "ARCHIVE"],
                    key=lambda x: -x["size_mb"])[:25]:
        print(f"  {r['size_mb']:>9.1f}MB  {r['path']:<45} {r['reason']}")

    print("\n# REVIEW 明细（需人工判断）:")
    for r in sorted([x for x in rows if x["action"] == "REVIEW"],
                    key=lambda x: -x["size_mb"])[:30]:
        print(f"  {r['size_mb']:>9.1f}MB  {r['path']:<45} {r['reason']}")


if __name__ == "__main__":
    main()
