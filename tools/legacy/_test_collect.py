import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "G:/GitHub/DiT/tools")
import pull_monitor as pm
result = pm.collect_remote()
print("num values:", len(result))
if result and result[0]:
    latest, log, eval_json, eval_state, eval_log, ckpt_dir, es, ss = result
    print("latest:", latest)
    print("ckpt_dir:", ckpt_dir)
    print("log lines:", len(log.splitlines()) if log else 0)
    print("eval_json length:", len(eval_json) if eval_json else 0)
    print("eval_json first 300:", repr(eval_json[:300]) if eval_json else "EMPTY")
    print("eval_samples:", es[:5])
    print("seen_samples:", ss[:5])
else:
    print("collect_remote returned None or empty")
