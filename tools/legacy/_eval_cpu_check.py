import psutil, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# check if eval workers are actually using CPU
for p in psutil.process_iter(["pid", "name"]):
    try:
        cl = p.cmdline()
        if cl and "auto_eval" in " ".join(cl) and "worker" not in " ".join(cl):
            print(f"Main eval PID={p.pid}")
            children = p.children(recursive=True)
            for c in children:
                cpu1 = c.cpu_percent(interval=0.5)
                mem = c.memory_info().rss / 1024 / 1024
                print(f"  child PID={c.pid} cpu={cpu1:.0f}% status={c.status()} mem={mem:.0f}MB")
            break
    except Exception:
        pass
# check overall CPU usage
print("\nCPU per core (top 10 busy):")
cpus = psutil.cpu_percent(percpu=True)
busy = sorted([(i, p) for i, p in enumerate(cpus)], key=lambda x: -x[1])[:10]
for i, p in busy:
    print(f"  core {i}: {p:.0f}%")
