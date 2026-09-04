import csv, os

path = "5script/all_experiments_eval_20260903.csv"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Total runs recorded: {len(rows)}")
    valid_rows = []
    for r in rows:
        val = r.get("best_ssim")
        if val and val not in ("None", "", "nan"):
            try:
                r["_ssim"] = float(val)
                valid_rows.append(r)
            except:
                pass
    valid_rows.sort(key=lambda x: x["_ssim"], reverse=True)
    print("\nTop 20 Experiments across All Generations:")
    for r in valid_rows[:20]:
        series = r.get("series", "")
        run = r.get("run", "")
        model = r.get("cfg_model", "")
        step = r.get("best_step", "")
        eval_csv = os.path.basename(r.get("cfg_eval_csv", ""))
        diff = r.get("cfg_diffusion_type", "")
        skel = "skel" if r.get("cfg_skel_latent_shards_dir") else "base"
        print(f"  [{series:<15s}] {run[:42]:<42s} | SSIM={r['_ssim']:.4f} | step={step:<7s} | {model:<14s} | {diff:<4s} | {skel:<4s} | eval={eval_csv}")

    print("\nSummary by Series:")
    series_map = {}
    for r in valid_rows:
        s = r.get("series", "other")
        series_map.setdefault(s, []).append(r["_ssim"])
    for s, scores in sorted(series_map.items()):
        print(f"  Series: {s:<20s} count={len(scores):<3d} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}")
