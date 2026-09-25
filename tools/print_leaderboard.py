import json

data = json.load(open("assets/archeology_survey.json", encoding="utf-8"))
valid = [e for e in data["experiments"] if e.get("best_strict")]
valid.sort(key=lambda x: x["best_strict"][0], reverse=True)
print(f"Total exps with strict SSIM: {len(valid)}")
print(f"{'Experiment Name':<45} | {'Step':<8} | {'Best Strict':<12} | {'Seen'}")
print("-" * 80)
for e in valid:
    st_val, st_step, _ = e["best_strict"]
    sn_val = round(e["best_seen"][0], 4) if e.get("best_seen") else "N/A"
    print(f"{e['name']:<45} | {str(st_step):<8} | {st_val:<12.4f} | {sn_val}")
