#!/opt/conda/envs/cu121/bin/python
"""Debug: check merged wild.zip"""
import os, zipfile
import pyzipper

WILD = "/root/Workspace/xy/HCSU/wild.zip"

print(f"File: {WILD}")
print(f"Size: {os.path.getsize(WILD)/1024/1024/1024:.1f}GB")

print("\n--- pyzipper test ---")
try:
    with pyzipper.AESZipFile(WILD, 'r') as z:
        names = z.namelist()
        print(f"File count: {len(names)}")
        print(f"First 10 names: {names[:10]}")
        # Try reading first non-directory file
        for name in names[:20]:
            if not name.endswith('/'):
                try:
                    data = z.read(name)
                    print(f"Read {name}: {len(data)} bytes, magic={data[:4].hex()}")
                    break
                except Exception as e:
                    print(f"Read {name}: {type(e).__name__}: {e}")
except Exception as e:
    print(f"Error opening: {type(e).__name__}: {e}")

print("\n--- standard zipfile test ---")
try:
    with zipfile.ZipFile(WILD, 'r') as z:
        names = z.namelist()
        print(f"File count: {len(names)}")
        print(f"First 10 names: {names[:10]}")
        for name in names[:20]:
            if not name.endswith('/'):
                try:
                    data = z.read(name)
                    print(f"Read {name}: {len(data)} bytes, magic={data[:4].hex()}")
                    break
                except Exception as e:
                    print(f"Read {name}: {type(e).__name__}: {e}")
except Exception as e:
    print(f"Error opening: {type(e).__name__}: {e}")
