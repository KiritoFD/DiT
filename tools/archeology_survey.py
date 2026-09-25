#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""archeology_survey.py — Survey all remote experiments, logs, and data."""

import os
import glob
import json
import csv

ROOT = "/root/Workspace/xy/DiT"

def inspect_results():
    results_dir = os.path.join(ROOT, "assets/results")
    if not os.path.exists(results_dir):
        print("Results dir not found: " + results_dir)
        return []

    experiments = []
    
    for entry in sorted(os.listdir(results_dir)):
        epath = os.path.join(results_dir, entry)
        if not os.path.isdir(epath) or entry.startswith("_"):
            continue
            
        exp_info = {
            "name": entry,
            "path": epath,
            "summary_csv": None,
            "metrics": {},
            "best_strict": None,
            "best_seen": None,
            "latest_step": None,
            "has_posters": os.path.exists(os.path.join(epath, "posters")),
            "resolved_cfg": None
        }

        # Check summary csv
        for sname in ["eval_stdskel_summary.csv", "eval_summary.csv", "summary.csv"]:
            scsv = os.path.join(epath, sname)
            if os.path.exists(scsv):
                exp_info["summary_csv"] = scsv
                try:
                    with open(scsv, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            exp_info["rows_count"] = len(rows)
                            stricts = []
                            seens = []
                            for r in rows:
                                step = r.get("step") or r.get("train_steps")
                                set_name = (r.get("set") or "").strip().lower()
                                val_str = r.get("ssim_mean") or r.get("ssim")
                                if set_name == "strict":
                                    if val_str:
                                        try: stricts.append((float(val_str), step, r))
                                        except: pass
                                elif set_name == "seen":
                                    if val_str:
                                        try: seens.append((float(val_str), step, r))
                                        except: pass
                                else:
                                    st = r.get("strict_ssim") or r.get("strict") or r.get("ssim_strict")
                                    sn = r.get("seen_ssim") or r.get("seen") or r.get("ssim_seen")
                                    if st:
                                        try: stricts.append((float(st), step, r))
                                        except: pass
                                    if sn:
                                        try: seens.append((float(sn), step, r))
                                        except: pass
                            if stricts:
                                stricts.sort(key=lambda x: x[0], reverse=True)
                                exp_info["best_strict"] = stricts[0]
                            if seens:
                                seens.sort(key=lambda x: x[0], reverse=True)
                                exp_info["best_seen"] = seens[0]
                            if rows:
                                exp_info["latest_row"] = rows[-1]
                except Exception as e:
                    exp_info["error_csv"] = str(e)
                break

        # Check resolved_config
        cfg_files = glob.glob(os.path.join(epath, "**/resolved_config.json"), recursive=True)
        if cfg_files:
            try:
                with open(cfg_files[0], "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    exp_info["config"] = {
                        "model": cfg_data.get("model"),
                        "fusion": cfg_data.get("condition_fusion"),
                        "deform": cfg_data.get("deform_skel"),
                        "lr": cfg_data.get("lr"),
                        "batch": cfg_data.get("global_batch_size"),
                        "w_repa": cfg_data.get("w_repa")
                    }
            except:
                pass

        experiments.append(exp_info)

    return experiments


def inspect_logs():
    logs_dir = os.path.join(ROOT, "logs")
    log_files = glob.glob(os.path.join(logs_dir, "**/*.log"), recursive=True)
    summary_logs = []
    for lf in sorted(log_files):
        sz = os.path.getsize(lf)
        rel_path = os.path.relpath(lf, ROOT)
        last_eval = None
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-150:]
                for line in reversed(lines):
                    if "[in-mem-eval]" in line and "step=" in line:
                        last_eval = line.strip()
                        break
        except:
            pass
        summary_logs.append({
            "path": rel_path,
            "size_kb": sz // 1024,
            "last_eval": last_eval
        })
    return summary_logs


def inspect_data():
    data_dir = os.path.join(ROOT, "data")
    data_info = {}
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                count = 0
                tot_sz = 0
                try:
                    for root, dirs, files in os.walk(item_path):
                        for f in files:
                            p = os.path.join(root, f)
                            count += 1
                            try:
                                tot_sz += os.path.getsize(p)
                            except (FileNotFoundError, OSError):
                                pass
                except Exception:
                    pass
                data_info[item] = {
                    "count": count,
                    "size_mb": round(tot_sz / 1024 / 1024, 2)
                }
    return data_info


if __name__ == "__main__":
    print("=== Remote Survey Start ===")
    exps = inspect_results()
    print("Found " + str(len(exps)) + " experiment directories in assets/results")
    
    valid_exps = [e for e in exps if e.get("best_strict")]
    valid_exps.sort(key=lambda x: x["best_strict"][0], reverse=True)
    
    print("\n--- Leaderboard by Best Strict SSIM ---")
    print(f"{'Experiment Name':<42} | {'Step':<8} | {'Best Strict':<12} | {'Seen':<8} | {'Model/Fusion'}")
    print("-" * 95)
    for e in valid_exps:
        b_st, step, r = e["best_strict"]
        b_sn = r.get("seen_ssim", r.get("seen", "N/A"))
        cfg_str = ""
        if e.get("config"):
            cfg_str = f"{e['config'].get('model', '')}/{e['config'].get('fusion', '')}"
        print(f"{e['name']:<42} | {str(step):<8} | {b_st:.4f}       | {str(b_sn)[:6]:<8} | {cfg_str}")

    logs = inspect_logs()
    print("\nFound " + str(len(logs)) + " log files. Notable evals:")
    for l in logs:
        if l["last_eval"]:
            print(f"  [{l['path']} ({l['size_kb']}KB)] -> {l['last_eval'][:100]}")

    data = inspect_data()
    print("\nData Assets Overview:")
    for k, v in sorted(data.items()):
        print(f"  {k}: {v['count']} files, {v['size_mb']} MB")

    out_json = os.path.join(ROOT, "assets/archeology_survey.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"experiments": exps, "logs": logs, "data": data}, f, indent=2, ensure_ascii=False)
    print("\nSurvey data written to: " + out_json)
