import inspect, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import diffusers.models.autoencoders.autoencoder_kl as mod
full = inspect.getsource(mod)
idx = full.find("def _decode")
while idx != -1:
    print("=== _decode at", idx, "===")
    print(full[idx:idx+2500])
    print()
    idx = full.find("def _decode", idx+1)
    if idx > 60000:
        break
