# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# 用 python 读远程 log, 找 50000 save 之后的内容
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "python3 -c \""
    "lines=open('/root/Workspace/xy/DiT/run_px_s.log',encoding='utf-8',errors='replace').readlines();"
    "for i,l in enumerate(lines):"
    "  if '0050000.pt' in l:"
    "    for j in range(i, min(i+8, len(lines))): print(j, lines[j].rstrip());"
    "    break"
    "\""],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}\n{r.stderr[:300]}")