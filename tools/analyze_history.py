import os
import sys
import glob
import csv
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_all_eval_csvs():
    summary_files = glob.glob(os.path.join(ROOT, "assets/results/**/eval_stdskel_summary.csv"), recursive=True)
    all_runs = {}
    for sf in summary_files:
        exp_dir = os.path.dirname(sf)
        exp_name = os.path.basename(exp_dir)
        # also check parent dir if exp_dir has timestamp
        if exp_name.startswith("2026"):
            exp_name = os.path.basename(os.path.dirname(exp_dir))
        
        with open(sf, "r", encoding="utf-8", errors="ignore") as f:
            reader = list(csv.DictReader(f))
        
        if not reader:
            continue
            
        steps_data = defaultdict(dict)
        for r in reader:
            step = r.get("step") or r.get("train_steps")
            if not step:
                continue
            try:
                step = int(step)
            except:
                continue
            set_name = (r.get("set") or "").strip().lower()
            val = float(r.get("ssim_mean") or r.get("ssim") or 0.0)
            mse = float(r.get("mse_mean") or r.get("mse") or 0.0)
            lpips = float(r.get("lpips_mean") or r.get("lpips") or 0.0)
            target_spec = r.get("target_spec") or r.get("tgt_spec")
            cal_enrich = r.get("cal_enrich") or r.get("enrichment")
            frag = r.get("frag")
            hole = r.get("hole")
            
            steps_data[step][set_name] = {
                "ssim": val,
                "mse": mse,
                "lpips": lpips,
                "target_spec": float(target_spec) if target_spec and target_spec != "None" else None,
                "cal_enrich": float(cal_enrich) if cal_enrich and cal_enrich != "None" else None,
                "frag": float(frag) if frag and frag != "None" else None,
                "hole": hole
            }
            
        # config if exists
        cfg_path = None
        for cf in glob.glob(os.path.join(exp_dir, "**/resolved_config.json"), recursive=True):
            cfg_path = cf
            break
        cfg = {}
        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as fp:
                    cfg = json.load(fp)
            except:
                pass
                
        all_runs[exp_name] = {
            "exp_name": exp_name,
            "dir": exp_dir,
            "steps": steps_data,
            "config": {
                "model": cfg.get("model"),
                "fusion": cfg.get("condition_fusion"),
                "lr": cfg.get("lr"),
                "batch": cfg.get("global_batch_size"),
                "deform_skel": cfg.get("deform_skel"),
                "w_repa": cfg.get("w_repa"),
                "w_deform_skel": cfg.get("w_deform_skel")
            }
        }
    return all_runs

if __name__ == "__main__":
    runs = parse_all_eval_csvs()
    print(f"Parsed {len(runs)} distinct runs with eval summaries.")
    
    # rank by best strict ssim
    ranked = []
    for k, v in runs.items():
        best_strict = None
        best_seen = None
        for step, sets in v["steps"].items():
            if "strict" in sets and sets["strict"]["ssim"] > 0:
                if best_strict is None or sets["strict"]["ssim"] > best_strict[0]:
                    best_strict = (sets["strict"]["ssim"], step, sets["strict"])
            if "seen" in sets and sets["seen"]["ssim"] > 0:
                if best_seen is None or sets["seen"]["ssim"] > best_seen[0]:
                    best_seen = (sets["seen"]["ssim"], step, sets["seen"])
        ranked.append((k, best_strict, best_seen, len(v["steps"]), v["config"]))
        
    ranked.sort(key=lambda x: x[1][0] if x[1] else 0, reverse=True)
    
    print("\nTop 20 runs by peak strict SSIM:")
    for r in ranked[:20]:
        st = f"{r[1][0]:.4f} (step {r[1][1]})" if r[1] else "None"
        sn = f"{r[2][0]:.4f} (step {r[2][1]})" if r[2] else "None"
        print(f"  {r[0]:<42} | Strict: {st:<22} | Seen: {sn:<22} | Steps count: {r[3]}")
        
    # save full structured output
    with open("assets/all_runs_deep_parsed.json", "w", encoding="utf-8") as f:
        # convert step keys to str
        serializable = {}
        for k, v in runs.items():
            serializable[k] = {
                "exp_name": v["exp_name"],
                "config": v["config"],
                "steps": {str(step): sdata for step, sdata in v["steps"].items()}
            }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print("\nFull parsed data saved to assets/all_runs_deep_parsed.json")
