import psutil, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for p in psutil.process_iter(["pid", "name", "status", "cmdline"]):
    try:
        cl = p.info.get("cmdline") or []
        if cl and "auto_eval" in " ".join(cl):
            mem = p.memory_info().rss / 1024 / 1024
            print(f"PID={p.info['pid']} status={p.info['status']} mem={mem:.0f}MB")
    except Exception:
        pass
# check children of main eval process
for p in psutil.process_iter(["pid", "name"]):
    try:
        if p.info.get("name") and "python" in p.info["name"].lower():
            cl = p.cmdline()
            if cl and "auto_eval" in " ".join(cl):
                children = p.children(recursive=True)
                print(f"  PID={p.pid} children={len(children)}")
                for c in children:
                    print(f"    child PID={c.pid} status={c.status()}")
    except Exception:
        pass
