# -*- coding: utf-8 -*-
"""查 50000 ckpt 保存后到 50020 之间的行 (eval 应该在这里运行但没输出)."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# eval 调用在 save_ckpt 之后, 如果 eval 抛异常应该有 warning
# 看看 log 里有没有任何 warning 行
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -n 'WARNING\\|warning\\|fail\\|Error\\|Traceback\\|eval-gpu' /root/Workspace/xy/DiT/run_px_s.log"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("WARNINGS:", r.stdout if r.returncode==0 else f"rc={r.returncode}")

# 看时间戳: 50000 save 在 15:15:26, 50020 在 15:15:31 -> 5秒间隔, eval 应该卡在这里
# 但没看到 eval 输出. 可能 eval 静默失败了. 看 50000.pt 之后的行号
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "awk '/Saved.*0050000/{found=1} found{print NR\": \"\\$0; count++} count>8{exit}' /root/Workspace/xy/DiT/run_px_s.log"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("AFTER 50k SAVE:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}")