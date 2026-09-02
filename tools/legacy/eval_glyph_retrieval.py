# -*- coding: utf-8 -*-
"""
eval_glyph_retrieval.py — 用现成 DINOv2 做「检索式字形准确率」评估（无需训练分类器）

动机
----
字形正确率是判断「base 0.50 到底是字形错还是细节糊」的关键指标，但训练
分类器在本数据上行不通：fame 只有 4,765 字 / 51,322 样本，每字平均 10.8
个样本，判别器必然欠拟合。

替代方案：直接用预训练 DINOv2 的 CLS 特征做最近邻检索。
  - 建库：train_fame 的图 → 每个 (script, character) 一个类中心
  - query：图 → DINO CLS → 在**同 script 组内**检索最近类中心
  - 指标：top-1 / top-5 字符准确率

为什么要按 script 分组检索
--------------------------
项目已有实测（src/train/train.py 注释）：DINO CLS 被「书体」主导，83% 的
最近邻是同一书体、跨书体字符检索 top-1 仅 1.9%。书体是混淆变量，必须先在
组内消除，剩下的才是纯字符身份判别力。

两个模式
--------
  raw      : CLS 特征 L2 归一化后直接检索
  centered : 先按 script 减去组内均值（去书体分量），再 L2 归一化 + 检索
             —— 与训练侧 dino_per_script_center 同源，验证它是否真的提升
                字符判别力（训练侧实测：有效秩 34.1→57.0）

校准先行（本脚本的核心用法）
----------------------------
先用 **真实图** 做 query（eval_fame_strict 的 500 张真实图，它们不在库里），
得到该方法的准确率上界。只有这个数够高，用它测出来的「生成图字形准确率」
才可信。真实图准确率 vs 生成图准确率 的差值 = 生成过程损失的字形信息。

用法（远程）
-----------
  python tools/eval_glyph_retrieval.py --build-csv 5script/train_fame.csv \
      --query-csv 5script/eval_fame_strict.csv --img-root final_imgs_256 \
      --cache dino_feat_cache.npz --mode both --limit 0
"""
import os, sys, csv, json, argparse, re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_id_of(row, col="image_path"):
    """从 image_path 提取数字 img_id（兼容 final_images/6.png 与 images/.../6.png）。"""
    m = re.search(r"(\d+)\.(png|jpg|jpeg)$", str(row.get(col, "")), re.I)
    if m:
        return int(m.group(1))
    v = row.get("img_id") or row.get("id")
    return int(v) if v else None


def load_dino(device):
    """加载本地 DINOv2（复用 src/loss/losses.py 的无网络加载路径）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.loss.losses import _load_local_dinov2, _default_dino_ckpt
    ckpt = _default_dino_ckpt()
    if not ckpt:
        raise RuntimeError("未找到本地 DINOv2 ckpt（pretrained_models/dinov2_vits14_pretrain.safetensors）")
    print(f"[dino] loading {ckpt}", flush=True)
    model = _load_local_dinov2(ckpt).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feats(model, rows, img_root, device, batch=128, tag="feat"):
    """对一组样本提取 DINO CLS 特征 (N, D)，L2 归一化。"""
    feats, keep = [], []
    bufs = []
    t0 = time.time()

    def flush():
        if not bufs:
            return
        x = torch.from_numpy(np.stack(bufs)).to(device)
        out = model(pixel_values=x)
        cls = out.last_hidden_state[:, 0]          # (B, D) CLS token
        cls = F.normalize(cls.float(), dim=-1)
        feats.append(cls.cpu().numpy())
        bufs.clear()

    skipped = 0
    for i, row in enumerate(rows):
        iid = img_id_of(row)
        path = os.path.join(img_root, f"{iid}.png") if iid is not None else None
        if path is None or not os.path.exists(path):
            skipped += 1
            continue
        try:
            im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            skipped += 1
            continue
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        bufs.append(a.transpose(2, 0, 1))
        keep.append(i)
        if len(bufs) >= batch:
            flush()
            if len(keep) % (batch * 20) == 0:
                el = time.time() - t0
                print(f"  [{tag}] {len(keep)}/{len(rows)} ({el:.0f}s, "
                      f"ETA {el/max(len(keep),1)*(len(rows)-len(keep)):.0f}s)", flush=True)
    flush()
    if not feats:
        raise RuntimeError(f"[{tag}] 没有提取到任何特征")
    F_all = np.concatenate(feats, axis=0)
    print(f"  [{tag}] feats={F_all.shape} skipped={skipped} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return F_all, np.array(keep, dtype=np.int64)


def build_centers(feats, rows, keep):
    """每个 (script, character) 一个类中心（该类所有样本特征的平均，再 L2 归一化）。"""
    groups = {}
    for j, i in enumerate(keep):
        r = rows[i]
        key = (r["script"], r["character"])
        groups.setdefault(key, []).append(feats[j])
    centers, labels, script_of = [], [], []
    for k, (sc, ch) in enumerate(sorted(groups)):
        v = np.stack(groups[(sc, ch)]).mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        centers.append(v)
        labels.append((sc, ch))
        script_of.append(sc)
    return np.stack(centers), labels, np.array(script_of)


def per_script_center(mat, scripts):
    """按 script 去均值 + L2 归一化（与训练侧 dino_per_script_center 同源）。"""
    out = mat.copy()
    for s in np.unique(scripts):
        m = scripts == s
        if m.sum() > 1:
            out[m] -= out[m].mean(0, keepdims=True)
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def retrieve(q, centers, script_of, labels, q_script):
    """在同 script 组内检索最近类中心，返回 (是否top1命中, 是否top5命中)。"""
    m = script_of == q_script
    if m.sum() == 0:
        return None, None
    sub = centers[m]
    sub_labels = [labels[i] for i in np.where(m)[0]]
    sim = sub @ q
    order = np.argsort(-sim)
    top1 = sub_labels[order[0]][1]
    top5 = {sub_labels[k][1] for k in order[:5]}
    return top1, top5


def run_eval(q_feats, q_rows, q_keep, centers, labels, script_of, mode_name):
    hit1 = hit5 = n = 0
    per_script = {}
    for j, i in enumerate(q_keep):
        r = q_rows[i]
        t1, t5 = retrieve(q_feats[j], centers, script_of, labels, r["script"])
        if t1 is None:
            continue
        n += 1
        ok1 = int(t1 == r["character"])
        ok5 = int(r["character"] in t5)
        hit1 += ok1
        hit5 += ok5
        d = per_script.setdefault(r["script"], [0, 0, 0])
        d[0] += 1
        d[1] += ok1
        d[2] += ok5
    print(f"\n=== [{mode_name}] 检索式字形准确率 (n={n}) ===")
    print(f"  top-1: {hit1/max(n,1)*100:.2f}%   top-5: {hit5/max(n,1)*100:.2f}%")
    print(f"  {'书体':<8}{'n':>6}{'top1%':>9}{'top5%':>9}")
    for s, (c, a, b) in sorted(per_script.items(), key=lambda x: -x[1][0]):
        print(f"  {s:<8}{c:>6}{a/c*100:>9.1f}{b/c*100:>9.1f}")
    return {"mode": mode_name, "n": n,
            "top1": hit1 / max(n, 1), "top5": hit5 / max(n, 1),
            "per_script": {s: {"n": c, "top1": a / c, "top5": b / c}
                           for s, (c, a, b) in per_script.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-csv", required=True, help="建库集 CSV（训练集真实图）")
    ap.add_argument("--query-csv", required=True, help="query 集 CSV")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--cache", default="dino_feat_cache.npz", help="特征缓存 npz")
    ap.add_argument("--mode", default="both", choices=["raw", "centered", "both"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="建库集截断（0=全量，调试用）")
    ap.add_argument("--query-limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="5script/glyph_retrieval_calib.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    def read(p):
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    build_rows = read(args.build_csv)
    query_rows = read(args.query_csv)
    if args.limit > 0:
        build_rows = build_rows[:args.limit]
    if args.query_limit > 0:
        query_rows = query_rows[:args.query_limit]
    print(f"build rows={len(build_rows)}  query rows={len(query_rows)}", flush=True)

    # ── 特征缓存（建库集特征可复用，换 query 时不必重算）───────────────────
    cache = {}
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
        print(f"[cache] loaded {args.cache}: {list(cache.keys())}", flush=True)

    need_model = ("b_feat" not in cache) or ("q_feat" not in cache) or \
                 (cache.get("q_n", np.array([-1]))[0] != len(query_rows))
    if need_model:
        model = load_dino(device)
        if "b_feat" not in cache:
            bf, bk = extract_feats(model, build_rows, args.img_root, device,
                                   args.batch, tag="build")
            cache["b_feat"], cache["b_keep"] = bf, bk
            cache["b_n"] = np.array([len(build_rows)])
        qf, qk = extract_feats(model, query_rows, args.img_root, device,
                               args.batch, tag="query")
        cache["q_feat"], cache["q_keep"] = qf, qk
        cache["q_n"] = np.array([len(query_rows)])
        np.savez(args.cache, **cache)
        print(f"[cache] saved {args.cache}", flush=True)
    else:
        print("[cache] reuse cached features (no DINO forward)", flush=True)

    b_feat, b_keep = cache["b_feat"], cache["b_keep"]
    q_feat, q_keep = cache["q_feat"], cache["q_keep"]

    centers, labels, script_of = build_centers(b_feat, build_rows, b_keep)
    print(f"\n库: {len(labels)} 个 (script, character) 类中心, "
          f"script={sorted(set(script_of.tolist()))}", flush=True)
    # query 侧 script（用于分组检索 + centered 去均值）
    q_script = np.array([query_rows[i]["script"] for i in q_keep])

    results = []
    if args.mode in ("raw", "both"):
        results.append(run_eval(q_feat, query_rows, q_keep,
                                centers, labels, script_of, "raw"))
    if args.mode in ("centered", "both"):
        bc = build_centers(per_script_center(b_feat, np.array(
            [build_rows[i]["script"] for i in b_keep])),
            build_rows, b_keep)[0]
        # 类中心也要按 script 去均值，与 query 侧同域
        bc = per_script_center(bc, script_of)
        qc = per_script_center(q_feat, q_script)
        results.append(run_eval(qc, query_rows, q_keep,
                                bc, labels, script_of, "centered"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"build_csv": args.build_csv, "query_csv": args.query_csv,
                   "n_build": len(build_rows), "n_query": len(query_rows),
                   "n_classes": len(labels), "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
