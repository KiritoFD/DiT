#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/add_stage_to_poster_tree.py
一键增量同步/添加新实验阶段到 assets/poster_tree/ 并自动重新生成全景 Poster。

使用示例：
  # 从 4090 服务器拉取 v70 step 5000 样本并更新海报：
  python tools/add_stage_to_poster_tree.py \
    --stage-id "07_v70_stdskel_5k" \
    --title "六阶段: v70 纯骨架+3x增广" \
    --sub "去除字表 / std skel 潜变量注入 / C2OT流匹配" \
    --badge "Step 5k (Live) | 纯骨架冷启动验证" \
    --badge-type "accent" \
    --remote "4090:/root/Workspace/xy/DiT/assets/results/v70_aug_sp_stdskel_c2ot/eval_samples_ctrl/step0005000/g"

  # 或者从本地已有的样本目录直接增量添加：
  python tools/add_stage_to_poster_tree.py \
    --stage-id "07_v70_stdskel_5k" \
    --title "六阶段: v70 纯骨架+3x增广" \
    --sub "去除字表 / std skel 潜变量注入 / C2OT流匹配" \
    --badge "Step 5k | 纯骨架演进验证" \
    --badge-type "accent" \
    --local-dir "path/to/samples"
"""
import os
import sys
import argparse
import json
import shutil
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TREE_DIR = "assets/poster_tree"


def main():
    parser = argparse.ArgumentParser(description="增量添加阶段到 poster_tree 并重新渲染海报")
    parser.add_argument("--stage-id", type=str, required=True, help="阶段文件夹名称，如 07_v70_stdskel_5k")
    parser.add_argument("--title", type=str, required=True, help="阶段展示大标题")
    parser.add_argument("--sub", type=str, default="", help="阶段技术简述")
    parser.add_argument("--badge", type=str, default="", help="徽章指标文本")
    parser.add_argument("--badge-type", type=str, default="default", choices=["input", "default", "warning", "success", "accent", "gt"])
    parser.add_argument("--remote", type=str, default=None, help="远端路径 (host:path)")
    parser.add_argument("--local-dir", type=str, default=None, help="本地样本路径")
    parser.add_argument("--pattern", type=str, default="g{idx}.png", help="原始文件名匹配模式，如 g{idx}.png")
    args = parser.parse_args()

    stage_dir = os.path.join(TREE_DIR, args.stage_id)
    os.makedirs(stage_dir, exist_ok=True)

    # 1. 写入 meta.json
    meta = {
        "title": args.title,
        "sub": args.sub,
        "badge": args.badge,
        "badge_type": args.badge_type
    }
    with open(os.path.join(stage_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 阶段元数据已更新: {os.path.join(stage_dir, 'meta.json')}")

    # 2. 读取列元数据
    meta_cols_path = os.path.join(TREE_DIR, "meta_columns.json")
    columns = []
    if os.path.exists(meta_cols_path):
        with open(meta_cols_path, "r", encoding="utf-8") as f:
            columns = json.load(f)

    # 3. 收集样本图片
    if args.remote:
        print(f"正在从远端拉取样本: {args.remote}")
        for col in columns:
            idx = col["idx"]
            remote_fn = args.pattern.format(idx=idx)
            remote_full = f"{args.remote}/{remote_fn}"
            local_target = os.path.join(stage_dir, f"{idx:02d}.png")
            cmd = ["scp", remote_full, local_target]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  • 拉取样本 #{idx} -> {idx:02d}.png 成功")
            except Exception:
                print(f"  ⚠ 样本 #{idx} 拉取失败 (将使用占位符)")

    elif args.local_dir:
        print(f"正在从本地复制样本: {args.local_dir}")
        for col in columns:
            idx = col["idx"]
            local_fn = args.pattern.format(idx=idx)
            src_full = os.path.join(args.local_dir, local_fn)
            local_target = os.path.join(stage_dir, f"{idx:02d}.png")
            if os.path.exists(src_full):
                shutil.copy2(src_full, local_target)
                print(f"  • 复制样本 #{idx} -> {idx:02d}.png 成功")
            else:
                print(f"  ⚠ 本地样本 #{idx} 未找到 (将使用占位符)")

    # 4. 调用渲染脚本自动重新生成全景海报
    print("\n正在调用本地渲染脚本自动重生成全景 Poster...")
    from render_poster_from_tree import render_poster
    out_path = render_poster(tree_dir=TREE_DIR)
    print(f"\n🎉 增量更新完毕！最新全景 Poster 已生成: {out_path}")


if __name__ == "__main__":
    main()
