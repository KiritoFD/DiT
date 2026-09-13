import re, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Check pull_monitor's eval parsing — does it understand the new eval JSON format?
# The new format: {"step": 1000, "mse": 0.329, "ssim": 0.076, "skel_iou": 0.162, ...}
# The old format: {"step": 1000, "mse": ..., "ssim": ...}
# pull_monitor.py merge_eval_json should handle both

# Read pull_monitor.py merge_eval_json
with open("G:/GitHub/DiT/tools/pull_monitor.py", encoding="utf-8") as f:
    content = f.read()

# Find merge_eval_json function
import re as _re
m = _re.search(r"def merge_eval_json.*?(?=\ndef )", content, _re.S)
if m:
    print(m.group()[:500])
