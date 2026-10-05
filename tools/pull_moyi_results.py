import os, sys, subprocess

base_dir = "exp/moyi_top10_rf"
os.makedirs(os.path.join(base_dir, "posters"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "eval_samples"), exist_ok=True)

print("1. Pulling posters and logs from ssh 48...")
subprocess.run(["scp", "48:/home/ds/Workspace/moyi/results/moyi_top10_rf/posters/*.png", f"{base_dir}/posters/"], check=True)
subprocess.run(["scp", "48:/home/ds/Workspace/moyi/results/moyi_top10_rf/log.txt", f"{base_dir}/logs/"], check=True)
subprocess.run(["scp", "48:/home/ds/Workspace/moyi/results/moyi_top10_rf/vocab.json", f"{base_dir}/"], check=True)
subprocess.run(["scp", "-r", "48:/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_samples_step20000/*", f"{base_dir}/eval_samples/"], check=True)

print("2. Pulling latest converged checkpoint (Step 25,000)...")
subprocess.run(["scp", "48:/home/ds/Workspace/moyi/results/moyi_top10_rf/checkpoints/moyi_0025000.pt", f"{base_dir}/checkpoints/"], check=True)

print("All Moyi results, images, and checkpoints pulled back successfully!")
