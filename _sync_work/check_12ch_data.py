"""核验 v12_12ch 用的 aux 数据是否与 px60 对齐。"""
import numpy as np, glob, csv, re, os
os.chdir("/root/Workspace/xy/DiT")


def ids(d):
    s = set()
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    for f in fs:
        s |= set(int(x) for x in np.load(f)["img_ids"])
    return s, len(fs)


can, ncan = ids("data/aux/aux_canny_latents_base")
sk, nsk = ids("data/aux/inst_skel_latents_px60")
rows = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
need = set(int(re.search(r"(\d+)", os.path.basename(r["image_path"])).group(1)) for r in rows)

print("canny(base) ids %d (%d shards)   inst_skel(px60) ids %d (%d shards)   px60 需要 %d"
      % (len(can), ncan, len(sk), nsk, len(need)))
print("canny   覆盖 px60: %.1f%%   缺失 %d" % (100 * len(can & need) / len(need), len(need - can)))
print("inst_skel 覆盖 px60: %.1f%%" % (100 * len(sk & need) / len(need)))
print("canny 有但 px60 不用: %d" % len(can - need))
print()

for d in ["data/aux/aux_canny_latents_base", "data/aux/inst_skel_latents_px60"]:
    f = sorted(glob.glob(os.path.join(d, "shard_*.npz")))[0]
    z = np.load(f)
    print("  %s : shape %s dtype %s" % (d, z["latents"].shape, z["latents"].dtype))
    z.close()

# 关键: canny 的 img_id 与 px60 的 img_id 是否指同一张图?
# 抽样比对: 取一个 id, 看 canny latent 的墨量是否与 inst_skel 一致量级
common = sorted(can & need)[:200]
print("\n抽样 %d 个共同 id 的 canny 墨量:" % len(common))


def fetch(d, idlist):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(f)
        ii = [int(x) for x in z["img_ids"]]
        for j, i in enumerate(ii):
            if i in idlist:
                m[i] = np.asarray(z["latents"][j], dtype=np.float32)
        z.close()
    return m


sel = set(common)
cm = fetch("data/aux/aux_canny_latents_base", sel)
sm = fetch("data/aux/inst_skel_latents_px60", sel)
import statistics
ci = [float((np.abs(v) > 0.5).mean()) for v in cm.values()]
si = [float((np.abs(v) > 0.5).mean()) for v in sm.values()]
print("  canny   |lat|>0.5 占比 med=%.4f  std=%.4f" % (statistics.median(ci), statistics.pstdev(ci)))
print("  inst_skel |lat|>0.5 占比 med=%.4f  std=%.4f" % (statistics.median(si), statistics.pstdev(si)))
