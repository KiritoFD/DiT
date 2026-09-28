# -*- coding: utf-8 -*-
"""numpy>=1.24 compatibility patches for the three repos (np.int/np.float/np.bool)."""
import os

ROOTS = ["/root/Workspace/xy/DiT/baseline/VQ-Font",
         "/root/Workspace/xy/DiT/baseline/DG-Font",
         "/root/Workspace/xy/DiT/baseline/FontDiffuser"]

REPL = [("dtype=np.int)", "dtype=int)"),
        ("dtype=np.int,", "dtype=int,"),
        ("np.float(", "float("),
        ("np.bool(", "bool("),
        ("np.int(", "int("),
        ("np.float)", "float)"),
        ("np.int)", "int)")]


def main():
    n = 0
    for root in ROOTS:
        for dirpath, _, files in os.walk(root):
            if ".git" in dirpath or "__pycache__" in dirpath:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                src = open(p, encoding="utf-8", errors="ignore").read()
                new = src
                for a, b in REPL:
                    new = new.replace(a, b)
                if new != src:
                    open(p, "w", encoding="utf-8").write(new)
                    print("patched", p)
                    n += 1
    print(f"patched {n} files")


if __name__ == "__main__":
    main()
