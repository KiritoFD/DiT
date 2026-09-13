import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "G:/GitHub/DiT/tools")
import pull_monitor as pm
result = pm.collect_remote()
latest, log, eval_json, eval_state, eval_log, ckpt_dir, es, ss = result
# The eval_json contains "EVAL\n{...}" — the part label leaked in
# Check if it starts with "EVAL"
print("eval_json starts with:", repr(eval_json[:30]))
# Check parts
SEP = "===_DSH_SEP_==="
combined = pm._ssh(
    f"echo '{SEP}LOG'; "
    f"cat /root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear/20260824-164822-s10-b4-grey-clear/log.txt 2>/dev/null | tail -1; "
    f"echo '{SEP}EVAL'; "
    f"cat /root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear/*/checkpoints/eval_auto_*.json 2>/dev/null; "
    f"echo '{SEP}END'",
    timeout=60)
parts = combined.split(SEP)
for i, p in enumerate(parts):
    print(f"part[{i}]: {repr(p[:100])}")
