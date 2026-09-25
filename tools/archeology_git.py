import subprocess
import json

def get_git_log():
    cmd = ["git", "log", "--reverse", "--pretty=format:%H|||%ad|||%s", "--date=iso"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    lines = res.stdout.strip().split("\n")
    commits = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|||")
        if len(parts) == 3:
            commits.append({
                "hash": parts[0][:7],
                "full_hash": parts[0],
                "date": parts[1],
                "msg": parts[2]
            })
    return commits

if __name__ == "__main__":
    commits = get_git_log()
    print(f"Total commits parsed: {len(commits)}")
    print(f"First commit: {commits[0]}")
    print(f"Latest commit: {commits[-1]}")
    
    # Save to json
    with open("assets/git_history_parsed.json", "w", encoding="utf-8") as f:
        json.dump(commits, f, indent=2, ensure_ascii=False)
    print("Saved to assets/git_history_parsed.json")
