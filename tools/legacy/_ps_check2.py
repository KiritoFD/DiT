import psutil, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parent = psutil.Process(1593272)
print("parent status:", parent.status(), "cpu:", parent.cpu_percent(interval=1))
children = parent.children(recursive=True)
print(f"children: {len(children)}")
for c in children:
    print(f"  PID={c.pid} status={c.status()} cpu={c.cpu_percent(interval=0.5):.0f}% mem={c.memory_info().rss/1024/1024:.0f}MB")
# Also check all python processes
pyths = [p for p in psutil.process_iter(["pid","name"]) if p.info.get("name") and "python" in p.info["name"].lower()]
print(f"\ntotal python processes: {len(pyths)}")
