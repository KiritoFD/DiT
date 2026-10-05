import subprocess

cmd = """ssh -o ConnectTimeout=5 ds@10.222.120.101 '
ps -ef | grep encode_top10_aug | grep -v grep || echo "ENCODE_NOT_RUNNING"
ls -lh /home/ds/Workspace/moyi/data/top10_style23/shards_img_aug/ 2>/dev/null || true
'"""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== STATUS ON 48 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
