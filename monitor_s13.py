"""Monitor remote s13 training every 20 minutes (sleep 1200s).

Each poll: SSH to remote, grab step/loss/early-stop/done status via base64
-encoded bash (avoids quoting issues), append a line to s13_monitor.log and
print to stdout (readable via job_output). Exits when training is Done or
the process died (confirmed by 2 consecutive polls).
"""
import base64
import datetime
import subprocess
import sys
import time

REMOTE = "root@10.176.54.17"
PORT = "36430"
INTERVAL = 1200  # 20 minutes
LOG_PATH = "s13_monitor.log"
CONFIG = "s13_3top30_dino_xs"

REMOTE_SCRIPT = r'''
cd /root/Workspace/xy/DiT
LOG=/tmp/s13_train.log
echo "STEP=$(grep -o 'step=0*[0-9]*' $LOG | tail -1 | tr -d 'step=')"
echo "LAST=$(tail -1 $LOG | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260)"
echo "ES=$(grep 'early-stop' $LOG | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200)"
echo "DONECOUNT=$(tail -3 $LOG | grep -c 'Done!')"
echo "ALIVE=$(pgrep -f 'train.py --config %CONFIG%' | wc -l)"
echo "EVALJSON=$(ls -t 5script/results/%CONFIG%/*/checkpoints/eval_auto_*.json 2>/dev/null | head -1)"
echo "EVALSIZE=$(ls -l 5script/results/%CONFIG%/*/checkpoints/eval_auto_*.json 2>/dev/null | awk '{print $5}')"
'''.replace("%CONFIG%", CONFIG)


def ssh_poll():
    """Run the remote poll script, return dict or None on total failure."""
    b64 = base64.b64encode(REMOTE_SCRIPT.encode("utf-8")).decode("ascii")
    cmd = [
        "ssh", "-p", PORT,
        "-o", "ConnectTimeout=60",
        "-o", "ServerAliveInterval=15",
        "-o", "StrictHostKeyChecking=no",
        REMOTE, f"echo {b64} | base64 -d | bash",
    ]
    for attempt in range(4):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                out = {}
                for line in r.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        out[k.strip()] = v.strip()
                return out
        except Exception as e:
            print(f"  [poll] ssh exception: {e}")
        time.sleep(15)
    return None


def format_poll(out):
    if out is None:
        return "POLL FAILED (ssh unreachable)"
    parts = []
    if out.get("STEP"):
        parts.append(f"step={out['STEP']}")
    if out.get("ALIVE") is not None:
        alive = "alive" if int(out["ALIVE"]) > 0 else "DEAD"
        parts.append(alive)
    if out.get("DONECOUNT") and int(out["DONECOUNT"]) > 0:
        parts.append("DONE")
    if out.get("ES"):
        parts.append(f"es={out['ES']}")
    if out.get("LAST"):
        parts.append(f"last={out['LAST']}")
    if out.get("EVALJSON"):
        parts.append(f"eval={out['EVALJSON']} ({out.get('EVALSIZE','?')}B)")
    return " | ".join(parts)


def main():
    print(f"[monitor] start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(interval={INTERVAL}s, remote={REMOTE}, config={CONFIG})", flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== monitor start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
                   f"(interval={INTERVAL}s) ===\n")
        logf.flush()

        dead_streak = 0
        while True:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out = ssh_poll()
            line = f"[{ts}] {format_poll(out)}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

            if out is not None:
                done = int(out.get("DONECOUNT", 0) or 0) > 0
                alive = int(out.get("ALIVE", 0) or 0) > 0
                if done or not alive:
                    dead_streak += 1
                    print(f"[monitor] {'DONE' if done else 'process dead'} "
                          f"confirmed ({dead_streak}/2), {2-dead_streak} more poll to exit", flush=True)
                    if dead_streak >= 2:
                        print(f"[monitor] exiting at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
                        logf.write(f"=== monitor exit {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
                        logf.flush()
                        return
                else:
                    dead_streak = 0
            else:
                dead_streak = 0  # ssh flakiness is not death

            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
