import psutil, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cpus = psutil.cpu_percent(percpu=True)
busy = [(i, p) for i, p in enumerate(cpus) if p > 5]
print(f"total {len(cpus)} cores, busy (>5%): {len(busy)}")
for i, p in busy:
    print(f"  core {i}: {p:.0f}%")
# 找连续空闲段
idle_cores = [i for i, p in enumerate(cpus) if p <= 5]
print(f"idle cores: {len(idle_cores)}")
if idle_cores:
    print(f"  idle: {idle_cores[:20]}{'...' if len(idle_cores)>20 else ''}")
# RAM
vm = psutil.virtual_memory()
print(f"RAM: {vm.used/1e9:.1f}G / {vm.total/1e9:.1f}G used, {vm.available/1e9:.1f}G available")
