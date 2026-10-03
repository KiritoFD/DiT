import os, glob

print("Checking available data directories on server:")
data_dirs = glob.glob("data/*")
for d in sorted(data_dirs):
    if os.path.isdir(d):
        num_files = len(os.listdir(d))
        print(f"  {d:35s}: {num_files} items")

print("\nChecking MCCD subdirectories:")
mccd_dirs = glob.glob("MCCD/*") + glob.glob("MCCD/*/*")
for d in mccd_dirs:
    if os.path.isdir(d):
        print(f"  {d}")
