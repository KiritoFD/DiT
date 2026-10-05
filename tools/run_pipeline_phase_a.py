#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline_phase_a.py — 自动化运行阶段 A 的 4 个现代组件排雷实验与严格评测

实验清单 (全部只跑 40,000 步，与 v61 @40k 的 Strict SSIM 0.5821 对齐比较):
  Exp 1: v61-S + RMSNorm
  Exp 2: v61-S + QK-Norm
  Exp 3: v61-S + 2D RoPE
  Exp 4: v61-S + SwiGLU

裁决原则：
  Strict SSIM 不跌 (>= 0.5815)，且能提升步速、显存或收敛曲线。
  胜出的组件打包成 Modern-Block-Best，为阶段 B (Exp 5: Time-Gate) 做准备。
"""

import json
import os
import re
import subprocess
import sys
import time

WORKSPACE = "/home/ds/Workspace/DiT"
PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
EXP_ROOT = "/home/ds/Workspace/DiT/experiments/ablation_phase_a"
CONFIGS_DIR = os.path.join(EXP_ROOT, "configs")
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
LOGS_DIR = os.path.join(EXP_ROOT, "logs")

BASELINE_SSIM_40K = 0.5821
THRESHOLD_SSIM = 0.5815

EXPERIMENTS = [
    {
        "id": "exp1_rmsnorm",
        "title": "Exp 1: v61-S + RMSNorm",
        "component": "RMSNorm",
        "config_file": os.path.join(CONFIGS_DIR, "exp1_rmsnorm.json"),
    },
    {
        "id": "exp2_qknorm",
        "title": "Exp 2: v61-S + QK-Norm",
        "component": "QK-Norm",
        "config_file": os.path.join(CONFIGS_DIR, "exp2_qknorm.json"),
    },
    {
        "id": "exp3_rope",
        "title": "Exp 3: v61-S + 2D RoPE",
        "component": "2D RoPE",
        "config_file": os.path.join(CONFIGS_DIR, "exp3_rope.json"),
    },
    {
        "id": "exp4_swiglu",
        "title": "Exp 4: v61-S + SwiGLU",
        "component": "SwiGLU",
        "config_file": os.path.join(CONFIGS_DIR, "exp4_swiglu.json"),
    },
]


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_train_log(log_path):
    steps_per_sec = []
    mems = []
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_sps = re.search(r"Steps/Sec:\s*([\d\.]+)", line)
            if m_sps:
                steps_per_sec.append(float(m_sps.group(1)))
            m_mem = re.search(r"Mem:\s*([\d\.]+)G", line)
            if m_mem:
                mems.append(float(m_mem.group(1)))
    avg_sps = (
        sum(steps_per_sec[-100:]) / len(steps_per_sec[-100:])
        if steps_per_sec
        else 0.0
    )
    peak_mem = max(mems) if mems else 0.0
    return avg_sps, peak_mem


def run_experiment(exp):
    exp_id = exp["id"]
    cfg_file = exp["config_file"]
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, f"{exp_id}_train.log")
    eval_dir = os.path.join(exp_dir, "eval_40k")
    metrics_file = os.path.join(eval_dir, "metrics_40k.json")

    log("=" * 75)
    log(f"开始执行实验: {exp['title']}")
    log(f"配置文件: {cfg_file}")
    log(f"运行目录: {exp_dir}")
    log(f"日志路径: {log_file}")
    log("=" * 75)

    # 1. 寻找是否已有训练好的 40k checkpoint
    ckpt_40k = None
    possible_ckpts = [
        os.path.join(exp_dir, "checkpoints", "0040000.pt"),
    ]
    # 查找子目录中的 checkpoints (train.py 创建的时间戳目录)
    for root, dirs, files in os.walk(exp_dir):
        if "0040000.pt" in files:
            ckpt_40k = os.path.join(root, "0040000.pt")
            break

    if ckpt_40k and os.path.exists(ckpt_40k):
        log(f"[跳过训练] 检测到 Step 40,000 Checkpoint 已存在: {ckpt_40k}")
    else:
        log("[训练启动] 开始训练至 40,000 步...")
        cmd_train = [
            PYTHON,
            "-u",
            os.path.join(WORKSPACE, "src/train/train.py"),
            "--config",
            cfg_file,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{WORKSPACE}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        with open(log_file, "w", encoding="utf-8") as f_out:
            p = subprocess.Popen(
                cmd_train,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=WORKSPACE,
            )

            # 监控直到训练完成
            while p.poll() is None:
                time.sleep(30)
                # 读取日志最后一行
                try:
                    with open(
                        log_file, "r", encoding="utf-8", errors="ignore"
                    ) as f_r:
                        lines = f_r.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "step=" in last_line:
                                log(f"  {exp_id}: {last_line}")
                except Exception:
                    pass

        rc = p.returncode
        if rc != 0:
            log(f"❌ 训练异常退出, returncode = {rc}! 请检查日志: {log_file}")
            return None

        log("[训练完成] 40,000 步训练正常结束!")

        # 重新定位 ckpt
        for root, dirs, files in os.walk(exp_dir):
            if "0040000.pt" in files:
                ckpt_40k = os.path.join(root, "0040000.pt")
                break

    if not ckpt_40k or not os.path.exists(ckpt_40k):
        log(f"❌ 未找到 40k Checkpoint: {ckpt_40k}!")
        return None

    # 2. 执行标准化 40k 严格评测
    if os.path.exists(metrics_file):
        log(f"[跳过评测] 检测到评测结果已存在: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        log(f"[评测启动] 正在评测 Checkpoint: {ckpt_40k}...")
        eval_script = os.path.join(
            EXP_ROOT, "../../tools/eval_ablation_model.py"
        )
        if not os.path.exists(eval_script):
            eval_script = os.path.join(
                WORKSPACE, "experiments/ablation_phase_a/eval_ablation_model.py"
            )

        cmd_eval = [
            PYTHON,
            "-u",
            eval_script,
            "--ckpt",
            ckpt_40k,
            "--out-dir",
            eval_dir,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{WORKSPACE}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = "0"

        eval_log = os.path.join(LOGS_DIR, f"{exp_id}_eval.log")
        with open(eval_log, "w", encoding="utf-8") as f_eval:
            r = subprocess.run(
                cmd_eval,
                stdout=f_eval,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=WORKSPACE,
            )

        if r.returncode != 0 or not os.path.exists(metrics_file):
            log(f"❌ 评测失败! 查看日志: {eval_log}")
            return None

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        log("[评测完成] 评测指标已计算并落盘!")

    # 3. 汇总数据
    avg_sps, peak_mem = parse_train_log(log_file)
    strict_ssim = metrics["strict"]["ssim_mean"]
    strict_lpips = metrics["strict"]["lpips_mean"]
    strict_skel = metrics["strict"]["skel_iou_mean"]
    seen_ssim = metrics["seen"]["ssim_mean"]
    gap_ssim = metrics["gap_ssim"]

    passed = strict_ssim >= THRESHOLD_SSIM

    result = {
        "exp_id": exp_id,
        "title": exp["title"],
        "component": exp["component"],
        "strict_ssim": strict_ssim,
        "strict_ssim_med": metrics["strict"]["ssim_med"],
        "strict_lpips": strict_lpips,
        "strict_skel_iou": strict_skel,
        "seen_ssim": seen_ssim,
        "gap_ssim": gap_ssim,
        "steps_per_sec": avg_sps,
        "samples_per_sec": avg_sps * 780,
        "peak_vram_gb": peak_mem,
        "passed": passed,
        "delta_ssim_vs_v61": strict_ssim - BASELINE_SSIM_40K,
        "ckpt_path": ckpt_40k,
    }

    log(f"--- 【{exp['title']} 战报】---")
    log(
        f"  Strict SSIM: {strict_ssim:.4f} (基准: {BASELINE_SSIM_40K:.4f}, 阈值: {THRESHOLD_SSIM:.4f}) -> {'✅ 达标' if passed else '❌ 未达标'}"
    )
    log(f"  Strict LPIPS: {strict_lpips:.4f}")
    log(f"  Strict Skel : {strict_skel:.4f}")
    log(f"  Seen SSIM   : {seen_ssim:.4f} (Gap: {gap_ssim:+.4f})")
    log(
        f"  速度 / 显存 : {avg_sps:.2f} step/s ({avg_sps*780:.1f} smp/s) | {peak_mem:.2f} GB"
    )
    log("-" * 75 + "\n")
    return result


def main():
    log("==================================================================")
    log("启动阶段 A：现代组件“体检排雷”全自动化流水线")
    log(f"基准参考: v61 @40k Strict SSIM = {BASELINE_SSIM_40K:.4f}")
    log(f"裁决门槛: Strict SSIM >= {THRESHOLD_SSIM:.4f}")
    log("==================================================================")

    all_results = []
    for exp in EXPERIMENTS:
        res = run_experiment(exp)
        if res:
            all_results.append(res)
            # 及时保存阶段成绩
            with open(
                os.path.join(EXP_ROOT, "scorecard_progress.json"), "w"
            ) as f:
                json.dump(all_results, f, indent=2)

    # 4. 生成最终大决算得分榜与 Modern-Block-Best 裁决报告
    scorecard_path = os.path.join(EXP_ROOT, "SCORECARD_PHASE_A.md")
    winners = []
    lines = [
        "# 阶段 A：现代组件“体检排雷”全景裁决计分板 (Modern-Block Ablation Scorecard)\n",
        f"> **评测基准**：对齐 v61 @40k 官方基线（Strict SSIM = `{BASELINE_SSIM_40K:.4f}`）  ",
        f"> **裁决准则**：Strict SSIM 不跌（>= {THRESHOLD_SSIM:.4f}），且综合考量步速、显存及拓扑保持。  ",
        f"> **生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}  \n",
        "---\n",
        "## 1. 核心实测指标全景天梯表\n",
        "| 实验代号 | 测试组件 | **Strict SSIM** (升) | **Δ vs v61** | **Strict LPIPS** (降) | **skel_iou** (升) | **Seen SSIM** | **Gap** | **步速 (smp/s)** | **显存 (GB)** | **裁决判定** |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in all_results:
        verdict = "🏆 **入选 (WINNER)**" if r["passed"] else "❌ **淘汰 (DROP)**"
        if r["passed"]:
            winners.append(r["component"])
        delta_str = f"{r['delta_ssim_vs_v61']:+.4f}"
        lines.append(
            f"| **{r['exp_id']}** | {r['component']} | **`{r['strict_ssim']:.4f}`** | `{delta_str}` | {r['strict_lpips']:.4f} | {r['strict_skel_iou']:.4f} | {r['seen_ssim']:.4f} | {r['gap_ssim']:+.4f} | {r['samples_per_sec']:.1f} | {r['peak_vram_gb']:.2f}G | {verdict} |"
        )

    lines.append("\n---\n")
    lines.append("## 2. 裁决总结与 Modern-Block-Best 构成方案\n")
    if winners:
        winner_str = " + ".join(winners)
        lines.append(f"- **胜出组件**：`{winner_str}`")
        lines.append(
            f"- **打包方案**：构建组合架构 **`Modern-Block-Best`**（包含 {winner_str}）。"
        )
        lines.append(
            "- **阶段 B 衔接**：下一阶段（Exp 5: v63-TimeGate）将直接以 `Modern-Block-Best` 为底盘，挂载时间步动态门控！"
        )
    else:
        lines.append(
            "- **胜出组件**：全部现代组件均未达到 >= 0.5815 门槛，退回经典架构底座。"
        )

    scorecard_content = "\n".join(lines)
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(scorecard_content)

    log("\n" + "=" * 75)
    log("阶段 A 全部 4 组排雷实验已全部完成！")
    log(f"裁决报告已保存至: {scorecard_path}")
    log("=" * 75)
    print("\n" + scorecard_content)


if __name__ == "__main__":
    main()
