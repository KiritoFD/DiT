# -*- coding: utf-8 -*-
"""从远程拉取当前实验 eval_csv 前 N 个样本的 GT 真值(canny/skel) + 保存本地 csv。
与远程 gen_canny_skel.py 产物完全一致(直接拷 final_canny/final_skeleton 原图)。
每次重建 remote_gt(实验切换会换 eval 样本), 使 GT 行与 eval_samples 同批。
EVAL_CSV 可从命令行参数指定, 默认 kailishu_eval.csv(当前 v3b 楷隶项目)。
"""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUSER = "root"
RHOST = "10.176.54.17"
RPORT = "36430"
RBASE = "/root/Workspace/xy/DiT"
LOCAL_GT_DIR = os.path.join(HERE, "remote_gt")
LOCAL_SHOW5_CSV = os.path.join(HERE, "show5_eval.csv")
EVAL_CSV = sys.argv[1] if len(sys.argv) > 1 else "kailishu_eval.csv"
N_SHOW = 5


def _scp(remote_path, local_path):
    ok = subprocess.run(
        ["scp", "-o", "ConnectTimeout=15", "-P", RPORT,
         f"{RUSER}@{RHOST}:{remote_path}", local_path],
        capture_output=True, timeout=60)
    return ok.returncode == 0


def main():
    # 取 eval csv 前 N+1 行（含表头）存到本地 show5_eval.csv
    cmd = f"head -{N_SHOW + 1} {RBASE}/{EVAL_CSV}"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", RPORT,
                        f"{RUSER}@{RHOST}", cmd], capture_output=True, text=True,
                       timeout=30, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not (r.stdout or "").strip():
        print(f"[gt-pull] 无法读取 eval csv {EVAL_CSV}")
        return False
    out_txt = r.stdout
    with open(LOCAL_SHOW5_CSV, "w", encoding="utf-8") as f:
        f.write(out_txt)
    # 解析 id
    reader = list(csv.DictReader(out_txt.splitlines()))
    ids = [os.path.basename(row["image_path"])[:-4] for row in reader[:N_SHOW]]
    print(f"[gt-pull] eval_csv={EVAL_CSV} show5 ids: {ids}")

    # 拉取 GT 真值(增量: 已存在跳过; 不 rmtree, 抗网络波动)
    os.makedirs(os.path.join(LOCAL_GT_DIR, "canny"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_GT_DIR, "skel"), exist_ok=True)
    ok_all = True
    for pid in ids:
        for sub, local in [("canny", "canny"), ("skeleton", "skel")]:
            rp = f"{RBASE}/final_{sub}/{pid}.png"
            lp = os.path.join(LOCAL_GT_DIR, local, f"{pid}.png")
            if os.path.exists(lp):
                continue
            if not _scp(rp, lp):
                print(f"[gt-pull] 拉取失败 {pid} {sub}")
                ok_all = False
    # 判断完整性: 所有 n 个 id 的 canny+skel 都已就绪才返回 True
    have = lambda d, pid: os.path.exists(os.path.join(LOCAL_GT_DIR, d, f"{pid}.png"))
    complete = all(have("canny", p) and have("skel", p) for p in ids)
    print(f"[gt-pull] eval_csv={EVAL_CSV} ids={ids} complete={complete} (GT -> {LOCAL_GT_DIR})")
    return ok_all and complete


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
