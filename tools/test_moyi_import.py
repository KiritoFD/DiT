import sys, os
base = "/home/ds/Workspace/moyi/ref/moyi"
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, "moyun"))

print("Testing imports for Moyun...")
try:
    import moyun_2
    print("✓ Successfully imported moyun_2!")
except Exception as e:
    print(f"✗ Failed to import moyun_2: {e}")
