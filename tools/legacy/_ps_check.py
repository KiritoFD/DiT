import psutil, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
    try:
        cl = p.info.get("cmdline") or []
        if cl and "vae_noise" in " ".join(cl):
            mem = p.info.get("memory_info")
            mem_mb = mem.rss / 1024 / 1024 if mem else 0
            print(f"PID={p.info['pid']} cpu={p.info.get('cpu_percent',0):.0f}% mem={mem_mb:.0f}MB cmd={' '.join(cl[:4])}")
    except Exception:
        pass
