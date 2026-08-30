# -*- coding: utf-8 -*-
"""
_scan_assets.py — 清点仓库的数据资产与模型产物，产出 data_assets.csv。

覆盖
----
1. 数据集 CSV：行数 / 书家数 / 字数 / 书体数 / 文件大小
2. latent / 骨架 / 图片目录：文件数 / 磁盘占用 / latent 维度
3. 标准字形库：覆盖书体与字数
4. 预训练权重与 embedding：维度 / 大小

用途：给 experiments.csv 补上「数据规模」维度 —— 因为不同实验用的数据集
（top6 / top30 / midclean / fame）难度差异极大，SSIM 跨数据集不可比。
"""
import os, sys, csv, json, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROWS = []


def add(**kw):
    kw.setdefault("notes", "")
    ROWS.append(kw)


def csv_stats(path, limit=None):
    """统计数据集 CSV：行数/书家数/字数/书体数。"""
    n = 0
    cals, chars, scripts = set(), set(), set()
    try:
        with open(path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            for row in rd:
                n += 1
                if limit and n >= limit:
                    break
                for k, v in (("calligrapher", cals), ("character", chars),
                             ("script", scripts)):
                    if k in row and row[k]:
                        v.add(row[k])
    except Exception as e:
        return None, f"ERR {e}"
    return {"n_rows": n, "n_calligraphers": len(cals), "n_characters": len(chars),
            "n_scripts": len(scripts), "cols": cols}, ""


def dir_stats(path, sample_npz=None):
    """目录：文件数 + 磁盘大小(MB)，可选读一个 npz 拿 latent 维度。"""
    if not os.path.isdir(path):
        return None
    cnt = 0
    size = 0
    for root, _, files in os.walk(path):
        for fn in files:
            cnt += 1
            try:
                size += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    info = {"n_files": cnt, "size_mb": round(size / 1024 / 1024, 1)}
    if sample_npz:
        p = os.path.join(path, sample_npz)
        if os.path.isfile(p):
            try:
                with np.load(p) as d:
                    info["keys"] = ",".join(d.files)
                    for k in ("latents",):
                        if k in d.files:
                            info["latent_shape"] = str(d[k].shape)
                            info["latent_dtype"] = str(d[k].dtype)
                    if "img_ids" in d.files:
                        info["n_ids_in_sample"] = int(len(d["img_ids"]))
            except Exception as e:
                info["npz_err"] = str(e)[:60]
    return info


def main():
    t0 = time.time()

    # ── 1. 数据集 CSV ─────────────────────────────────────────────────────
    csvs = sorted(set(glob.glob("5script/*.csv")) |
                  set(glob.glob("*.csv")) | set(glob.glob("configs/*.csv")))
    for p in csvs:
        st, err = csv_stats(p)
        if not st:
            add(path=p, category="csv", notes=err)
            continue
        size_mb = round(os.path.getsize(p) / 1024 / 1024, 2)
        add(path=p, category="dataset_csv", n_items=st["n_rows"],
            n_calligraphers=st["n_calligraphers"], n_characters=st["n_characters"],
            n_scripts=st["n_scripts"], size_mb=size_mb,
            notes="cols=" + "|".join(st["cols"][:8]))

    # ── 2. latent / 骨架 / 图片目录 ───────────────────────────────────────
    groups = [
        ("image_dir", ["final_imgs_256", "final_images", "final_imgs"]),
        ("latent_dir", ["final_latents_fame", "final_latents_f4",
                        "final_latents_mid_clean", "final_latents"]),
        ("skel_png_dir", ["final_skel3_fame", "final_skel1_fame",
                          "final_skel3_mid_clean", "final_skel1"]),
        ("skel_latent_dir", ["final_skel_latents_fame", "final_skel_latents_fame_1px",
                             "final_skel_latents_mid_clean",
                             "final_skel_latents_train_1px",
                             "final_skel_latents_eval_1px"]),
    ]
    for cat, paths in groups:
        for p in paths:
            st = dir_stats(p, "shard_0000.npz")
            if st:
                add(path=p, category=cat, n_items=st["n_files"],
                    size_mb=st["size_mb"],
                    notes=f"latent_shape={st.get('latent_shape','')} "
                          f"dtype={st.get('latent_dtype','')} "
                          f"keys={st.get('keys','')}")

    # ── 3. 标准字形库 ─────────────────────────────────────────────────────
    for lib in ["src/utils/std_glyph_latent", "src/utils/std_glyph_latent_v2"]:
        if not os.path.isdir(lib):
            add(path=lib, category="glyph_lib", n_items=0, notes="** 目录不存在 **")
            continue
        fonts = {}
        for font in sorted(os.listdir(lib)):
            d = os.path.join(lib, font)
            if os.path.isdir(d):
                fonts[font] = len([f for f in os.listdir(d) if f.endswith(".npy")])
        add(path=lib, category="glyph_lib", n_items=sum(fonts.values()),
            size_mb=round(sum(os.path.getsize(os.path.join(lib, f, n))
                              for f in fonts for n in os.listdir(os.path.join(lib, f))
                              if n.endswith(".npy")) / 1024 / 1024, 1),
            notes="; ".join(f"{k}={v}" for k, v in fonts.items()))

    # ── 4. 预训练 / embedding ─────────────────────────────────────────────
    for p in ["pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy",
              "pretrained_models/dino_embeddings/glyph_dino_index.json",
              "pretrained_models/dinov2_vits14_pretrain.safetensors",
              "pretrained_models/sd-vae-ft-ema"]:
        if os.path.isfile(p):
            info = {"n_items": 1, "size_mb": round(os.path.getsize(p) / 1024 / 1024, 1)}
            if p.endswith(".npy"):
                try:
                    a = np.load(p)
                    info["notes"] = f"shape={a.shape} dtype={a.dtype}"
                except Exception as e:
                    info["notes"] = f"ERR {e}"[:60]
            elif p.endswith(".json"):
                try:
                    j = json.load(open(p, encoding="utf-8"))
                    if isinstance(j, dict):
                        info["notes"] = "keys=" + ",".join(list(j.keys())[:6])
                        for k, v in j.items():
                            if isinstance(v, list):
                                info["n_items"] = len(v)
                                break
                except Exception as e:
                    info["notes"] = f"ERR {e}"[:60]
            else:
                info["notes"] = "dir/file"
            add(path=p, category="pretrained", **info)
        elif os.path.isdir(p):
            add(path=p, category="pretrained", n_items=1, notes="VAE dir")

    # ── 写出 ──────────────────────────────────────────────────────────────
    cols = ["path", "category", "n_items", "n_calligraphers", "n_characters",
            "n_scripts", "size_mb", "notes"]
    os.makedirs("5script", exist_ok=True)
    with open("5script/data_assets.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print(f"# wrote 5script/data_assets.csv ({len(ROWS)} rows) in {time.time()-t0:.0f}s")

    print("\n# dataset_csv 摘要（按书家数）:")
    for r in sorted([x for x in ROWS if x["category"] == "dataset_csv"],
                    key=lambda x: -(x.get("n_calligraphers") or 0)):
        print(f"  {r['path']:<42} rows={r['n_items']:<8} cal={r['n_calligraphers']:<6} "
              f"char={r['n_characters']:<7} script={r['n_scripts']}")

    print("\n# 关键目录:")
    for r in ROWS:
        if r["category"] in ("latent_dir", "skel_latent_dir", "glyph_lib"):
            print(f"  [{r['category']:<15}] {r['path']:<38} "
                  f"n={r.get('n_items', 0):<8} {r.get('size_mb', 0)}MB  "
                  f"{r.get('notes', '')[:60]}")


if __name__ == "__main__":
    main()
