#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_fs_topic.py — 一条命令做一个 few-shot 新书家主题（选题→图片→latent→风格行→配置→自检）。

用法（远端）:
    PY=/opt/conda/envs/cu121/bin/python
    $PY _sync_work/build_fs_topic.py --dry-run 沈周-行 伊秉绶-行   # 只体检不写盘
    $PY _sync_work/build_fs_topic.py            沈周-行 伊秉绶-行   # 真正构建

产出（<t> = 书家-书体）:
    data/50k/fs6_imgs_<t>/                    256 图，**逐图**做过背景极性矫正
    assets/fs6_<t>_train.csv / _eval.csv      50/100，按"字"互斥，img_id 用独立号段
    data/50k/shards_fs6_<t>/                  img latent + names
    data/50k/shards_fs6_<t>_std/              骨架 latent + names   ← 旧管线缺的就是这一步
    assets/fs6_row_<t>.pt                     该主题自己的 DINO K-Means(K=4) 质心行 (1,1536)
    assets/callig_script_id_map_fs6_<t>.json  87 pair -> 88 pair
    src/train/configs/v15_fs6_<t>.json        配置

四条硬规矩（每条都对应 docs/system/74 §4 里一个作废过整轮实验的坑）:
  1) **号段隔离**: few-shot 的 img_id 从 900000 起。旧管线用 0..149，与 50k 的
     shards_std 号段重叠 -> 骨架条件查到的是**别的字**(沈周"时"配到王献之"㑺")，
     全程不报错，只是 loss 不降。
  2) **latent 自带 names**: npz 里写 names(源图名)，训练侧 (latent_dataset) 会逐行
     校验 names == CSV 该行 -> 张冠李戴直接拒绝启动，而不是静默训一整晚。
  3) **骨架同书体**: 同一个字在 楷/行/隶 下有多种 std 骨架(4536/5821 字有多种)，
     必须借**主题书体**那一本。
  4) **白底墨字**: 模型只见过白底；黑底(拓片/反相扫描)必须逐图反相后再编码。
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

WILD = os.environ.get("FS_SRC", "/root/Workspace/xy/HCSU/wild_extract")
CSV50K = "assets/train_50k_v2.csv"
MAP50K = "assets/callig_script_id_map.json"
BASE_CFG = "src/train/configs/v15a_multistyle_k4_pool.json"
PRE_TABLE = "assets/multistyle_k4_pretrained.pt"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
CK50K_STD = "data/50k/shards_std"

SCRIPT_ID = {"楷": "0", "行": "3", "隶": "4"}     # v15a 词表里只有这三种
NEW_RAW_ID = "9999"
SEED = 20260921
N_TRAIN, N_EVAL = 50, 100
ID_BASE, ID_SLOT_STRIDE = 900000, 1000
TOPIC_ORDER = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "宋高宗-楷", "徐渭-行",
               # MCCD four_attribute_dataset 里 50k 未训的楷书新书家（export_mccd_kai.py 导出）
               "张即之-楷", "郑道昭-楷", "王知敬-楷", "张裕钊-楷", "吴彩鸾-楷"]

# few-shot 只有 1536 个可训参数: 短跑不要 EMA(评测会看不见更新)、不要 WD、不要 REPA
# (REPA 的 DINO 缓存按 img_id 查表，号段一撞就是错特征)。
FS_OVERRIDE = {
    "w_repa": 0.0,
    "weight_decay": 0.0,
    "use_ema": False,
    "style_anchor_weight": 0.0,       # 锚的是旧 87 行，梯度被 hook 清零 -> 只剩污染 loss
    "cond_drop_all_prob": 0.0,        # few-shot 要考"给定风格能不能画对"，不该丢条件
    "lr": 1e-3,
    "lr_schedule": "constant",
    "warmup_steps": 50,
    "global_batch_size": N_TRAIN,     # 50 张 = 全批
    "max_steps": 150000 + 2000,
    "ckpt_every": 250,
    "epoch_steps": 250,
    "gpu_eval_every": 250,
}


ARM_OFFSET = {50: 0, 100: 1, 200: 2}


def parse_topic(t):
    """'沈周-行' -> ('沈周-行', 50)；'沈周-行@200' -> ('沈周-行', 200)。

    样本量 scaling 必须**嵌套**：train_50 ⊂ train_100 ⊂ train_200，且 eval 集一字不换，
    否则"50 张够不够"和"换了字"两件事分不开。所以 @N 只往 train 侧加字
    （从 usable[150:] 取），eval 永远是 usable[50:150]。
    """
    base, _, n = t.partition("@")
    return base, (int(n) if n else N_TRAIN)


def topic_slot(topic):
    """号段基址。老 50 臂保持原 slot(0..5) 不变，免得已跑过的数据作废；
    scaling 臂挪到 50+ (id 950000+)，与 seen 对照(920000)互不撞号。"""
    base, n = parse_topic(topic)
    if base not in TOPIC_ORDER:
        raise SystemExit(f"主题 {base} 不在 TOPIC_ORDER 里 —— 号段基址会不确定，"
                         f"已登记的顺序: {TOPIC_ORDER}")
    if n not in ARM_OFFSET:
        raise SystemExit(f"样本量臂只支持 {sorted(ARM_OFFSET)}，给了 {n}")
    bi = TOPIC_ORDER.index(base)
    return bi if n == N_TRAIN else 50 + bi * 3 + ARM_OFFSET[n]


# ── 50k 反查表 ───────────────────────────────────────────────────────────────
_std_by_char_script = None
_trained_names = None


def std_index():
    """(字, 书体id) -> (std_path, 该 50k 行号, character_id, glyph_id)。同键多行取 std 名最小者。"""
    global _std_by_char_script
    if _std_by_char_script is None:
        best = {}
        for idx, r in enumerate(csv.DictReader(open(CSV50K, encoding="utf-8"))):
            key = (r["character"], r["script_id"])
            sp = r.get("std_path") or ""
            if not sp or not os.path.isfile(sp):
                continue
            if key not in best or sp < best[key][0]:
                best[key] = (sp, idx, r["character_id"], r["glyph_id"])
        _std_by_char_script = best
        print(f"[50k] 有可用 std 的 (字,书体) 组合: {len(best):,}")
    return _std_by_char_script


def trained_names():
    global _trained_names
    if _trained_names is None:
        m = json.load(open(MAP50K, encoding="utf-8"))
        id2name = {}
        for r in csv.DictReader(open(CSV50K, encoding="utf-8")):
            id2name.setdefault(r["calligrapher_id"], r["calligrapher"])
        # 异体字归一: 文徵明 == 文征明（同一人不同写法，不能当全新书家）
        variant = {"徵": "征"}
        norm = lambda s: "".join(variant.get(c, c) for c in s)
        _trained_names = {norm(id2name.get(k.split(":")[0], "?"))
                          for k in m["pair_map"]}
    return _trained_names


# ── 选题 ─────────────────────────────────────────────────────────────────────
def _border_med(a, w=8):
    """图框 8px 边的中位数 = 纸/底的颜色（书法扫描件的字背景一定贴边）。"""
    ring = np.concatenate([a[:w].ravel(), a[-w:].ravel(), a[:, :w].ravel(), a[:, -w:].ravel()])
    return float(np.median(ring))


def _otsu(a):
    """Otsu 阈值（无魔数、内容自适应）。a: uint8 (H,W)。"""
    h = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    tot = h.sum()
    sum_all = (np.arange(256) * h).sum()
    w0 = s0 = 0.0
    best, thr = -1.0, 127
    for t in range(256):
        w0 += h[t]
        if w0 <= 0 or w0 >= tot:
            continue
        s0 += t * h[t]
        m0, m1 = s0 / w0, (sum_all - s0) / (tot - w0)
        between = w0 * (tot - w0) * (m0 - m1) ** 2
        if between > best:
            best, thr = between, t
    return thr


def _binarize(a, min_area=30):
    """灰度 -> {0,255} 二值图（与 50k 同口径），并去掉小于 min_area 的孤立墨点。

    去麻点很关键: 傅山-行是绢本，经纬织纹在 Otsu 之后会变成成片小墨点，
    实测该主题 Diff 卡在 0.55-0.63(其它主题 0.29-0.35)、eval 完全不动 ——
    模型拟合不了这种噪点目标，风格行学到的也是噪点。
    ⚠ 连通域必须标在**墨**上 (a <= thr)：早先写成标在亮域上，结果删的是"小块背景"、
      反而把麻点连成了片（ink 0.226 -> 0.228 不降反升，就是这个反向 bug）。
    """
    thr = _otsu(a)
    ink = a <= thr
    if min_area:
        from scipy import ndimage
        lab, n = ndimage.label(ink)
        if n:
            sizes = ndimage.sum(ink, lab, index=np.arange(1, n + 1))
            ink = np.isin(lab, np.where(sizes >= min_area)[0] + 1)
    return np.where(ink, 0, 255).astype(np.uint8)


def _prep(src, size=256):
    """原图 -> (是否反相, 256 二值图 uint8)。极性用贴边底色判，判不出就丢。"""
    from PIL import Image
    a = np.asarray(Image.open(src).convert("L"), dtype=np.uint8)
    b0 = _border_med(a)
    if abs(b0 - 128.0) < 12.0:
        raise ValueError(f"底色中位数 {b0:.0f} 落在 128±12 —— 极性判不出")
    if b0 < 128.0:
        a = 255 - a
    im = Image.fromarray(a).resize((size, size), Image.LANCZOS)
    return b0 < 128.0, _binarize(np.asarray(im, np.uint8))


def assess(src):
    """读一张 wild 原图 -> (可用?, 是否要反相, 二值化后墨占比)。

    极性判据用**贴边背景**而不是全图中位数：宋高宗·楷是绢本灰底(底≈130)+浓墨，
    全图中位数会掉到 128 以下 -> 用中位数判"黑底"会把**本来就是白底墨字**的图反相，
    得到"深底白字"的垃圾样本（实测 150 张里错了 109 张）。贴边中位数量的是底色。
    """
    try:
        inv, b = _prep(src)
    except Exception:
        return False, False, 0.0
    ink = float((b == 0).mean())
    return 0.05 <= ink <= 0.45, inv, ink


def select(topic):
    base, n_train = parse_topic(topic)
    cal, _, sc = base.partition("-")
    if sc not in SCRIPT_ID:
        raise SystemExit(f"[{topic}] 书体 {sc} 不在 v15a 词表 {list(SCRIPT_ID)} 里 —— 分布外")
    if cal in trained_names():
        raise SystemExit(f"[{topic}] 书家 {cal} **已在 87-pair 训练表里** —— "
                         f"不是全新书家，few-shot 结论无意义（异体字已归一）")
    sid = SCRIPT_ID[sc]
    folder = os.path.join(WILD, f"{cal}-{sc}")
    if not os.path.isdir(folder):
        raise SystemExit(f"[{topic}] 找不到 {folder}")
    files = {os.path.splitext(f)[0]: os.path.join(folder, f)
             for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    idx = std_index()
    need = n_train + N_EVAL          # train 与 eval 按字互斥，且 eval 恒为 cand[50:150]
    cand, rejected = [], 0
    for ch, path in sorted(files.items()):
        if (ch, sid) not in idx:
            continue
        ok, inv, ink = assess(path)
        if ok:
            cand.append((ch, inv, ink))
        else:
            rejected += 1
    if len(cand) < need:
        raise SystemExit(f"[{topic}] 过质检的字只有 {len(cand)} < {need} "
                         f"(被剔 {rejected} 张: 底色不够白/墨量异常/无同书体 std)")
    # ★ 种子只跟"基础主题"绑定，不随 @N 变 -> 各样本量臂的 cand 顺序一致，
    #   于是 train_50 ⊂ train_100 ⊂ train_200 且 eval 完全相同。
    rng = random.Random(SEED + TOPIC_ORDER.index(base))
    rng.shuffle(cand)
    chars = [c[0] for c in cand]
    tr = chars[:N_TRAIN] + chars[N_TRAIN + N_EVAL:N_TRAIN + N_EVAL + (n_train - N_TRAIN)]
    ev = chars[N_TRAIN:N_TRAIN + N_EVAL]
    assert not (set(tr) & set(ev)), "train/eval 必须按字互斥"
    assert len(set(tr)) == len(tr) == n_train, f"train 应有 {n_train} 个字"
    inv_n = sum(1 for c in cand[:need] if c[1])
    print(f"[质检] 候选 {len(cand)}/{len(files)} 张过闸(剔 {rejected})；本臂 train={n_train} "
          f"eval={N_EVAL}，选中的 {need} 张里 {inv_n} 张需反相")
    return cal, sc, sid, {c: files[c] for c in tr}, {c: files[c] for c in ev}


# ── 图片（极性矫正 + 二值化 + 256） ───────────────────────────────────────────
def fix_image(src, dst):
    """落盘一张与 50k 同口径的 256 图 -> (是否反相, 灰度均值, 墨占比, ==255 占比)

    二值化是必须的: 实测 50k 训练图的像素分布是**严格二值**的
    (80.7% 恰为 255, 19.1% 在 0-10, 中间几乎为空)。few-shot 若交灰底扫描件,
    模型要学的目标分布它从没见过 -> Diff 降不下去, 而且那 1536 个可训参数
    会去吸收"纸张灰度/噪点"这种与风格无关的 nuisance。
    """
    from PIL import Image
    inverted, b = _prep(src)
    im = Image.fromarray(b).convert("RGB")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst)
    return inverted, float(b.mean()), float((b == 0).mean()), float((b == 255).mean())


# ── latent 编码（与 50k 完全同口径: _tf_gray + VAEIO） ────────────────────────
_IO = None


def vae_io():
    global _IO
    if _IO is None:
        from src.data.vae_io import VAEIO
        _IO = VAEIO(VAE, device=os.environ.get("FS_DEVICE", "cuda"))
    return _IO


def encode_shard(paths, ids, names, out_dir, tag, batch=32):
    from src.data.vae_io import _tf_gray
    io = vae_io()
    lat = []
    for i in range(0, len(paths), batch):        # 一次全推会 150×256×256 把 4090 打爆
        xs = torch.stack([torch.from_numpy(_tf_gray(p)) for p in paths[i:i + batch]])
        lat.append(np.asarray(io.encode(xs)))
    lat = np.concatenate(lat, 0)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "shard_00000.npz")
    np.savez_compressed(out, latents=lat.astype(np.float16),
                        img_ids=np.array(ids, dtype=np.int64),
                        names=np.array(names, dtype="U64"))
    s = lat.astype(np.float32)
    print(f"[lat:{tag}] {len(paths)} 张 -> {out}  std={s.std():.3f} "
          f"mean={s.mean():+.3f} |abs|={np.abs(s).mean():.3f}")
    return out


# ── 该主题自己的 DINO K-Means 风格行（与建表同款构造） ────────────────────────
def dino_row(paths, out_pt, k=4):
    from src.loss.losses import _default_dino_ckpt, _load_local_dinov2
    from tools.build_multistyle_k4 import kmeans_torch, l2n
    import torch.nn.functional as F
    dev = os.environ.get("FS_DEVICE", "cuda")
    model = _load_local_dinov2(_default_dino_ckpt()).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            g = [np.asarray(_pil(p).convert("L"), dtype=np.float32)
                 for p in paths[i:i + 32]]
            t = torch.from_numpy(np.stack(g))[:, None]                    # (B,1,H,W)
            t = F.interpolate(t, size=(256, 256), mode="bicubic", align_corners=False)
            t = (t / 255.0).repeat(1, 3, 1, 1).to(dev)
            out = model((t - mean) / std)
            cls = (out["x_norm_clstoken"] if isinstance(out, dict) and "x_norm_clstoken" in out
                   else out.last_hidden_state[:, 0, :])
            feats.append(cls.float().cpu().numpy())
    feat = np.concatenate(feats)
    uniq = len(np.unique(np.round(feat, 4), axis=0))
    if uniq < max(10, len(feat) // 2):
        raise SystemExit(f"[dino] ✗ {len(feat)} 张图只有 {uniq} 个唯一特征 —— 提取管线坏了"
                         f"（历史教训: 漏 /255 -> ViT 饱和 -> CLS 恒定）")
    ft = torch.from_numpy(l2n(feat.astype(np.float32)))
    cent, inertia = kmeans_torch(ft, k, seed=SEED)
    emb = cent.reshape(1, -1)
    table = torch.load(PRE_TABLE, map_location="cpu", weights_only=False)["embedding"]
    print(f"[dino] feat {feat.shape} 唯一 {uniq} | K={k} inertia={inertia:.2f} | "
          f"新行范数={float(emb.norm()):.3f} vs 表内范数 "
          f"med={float(table.norm(dim=1).median()):.3f} "
          f"token 间 cos={F.cosine_similarity(cent.unsqueeze(0), cent.unsqueeze(1))[0, 0]:.3f}")
    torch.save({"embedding": emb, "centroids": cent.unsqueeze(0), "k_clusters": k,
                "dim": cent.shape[1], "n_src_imgs": len(paths), "inertia": inertia,
                "src": "few-shot 主题自己的 50 张 train 图（不含 eval）"}, out_pt)
    return out_pt


def _pil(p):
    from PIL import Image
    return Image.open(p)


# ── pair 词表 + 配置 ──────────────────────────────────────────────────────────
def write_map(topic, sid):
    base = json.load(open(MAP50K, encoding="utf-8"))
    key = f"{NEW_RAW_ID}:{sid}"
    if key in base["pair_map"]:
        raise SystemExit(f"[{topic}] pair {key} 已存在")
    m = json.loads(json.dumps(base))
    new_pair = int(m["num_pairs"])
    m["pair_map"][key] = new_pair
    m["callig_map"][NEW_RAW_ID] = int(m["num_calligraphers"])
    m["num_pairs"] = new_pair + 1
    m["num_calligraphers"] = int(m["num_calligraphers"]) + 1
    if isinstance(m.get("pair_to_callig"), list):
        m["pair_to_callig"].append(int(m["num_calligraphers"]) - 1)
    out = f"assets/callig_script_id_map_fs6_{topic}.json"
    json.dump(m, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out, new_pair


def write_config(topic, new_pair, map_path):
    d = json.loads(json.dumps(json.load(open(BASE_CFG, encoding="utf-8"))))
    d["experiment_name"] = f"v15-fs6-{topic}"
    d["results_dir"] = f"assets/results/v15_fs6_{topic}"
    d["num_calligraphers"] = new_pair + 1
    d["callig_script_map"] = map_path
    d["callig_emb_pretrained"] = PRE_TABLE
    d["init_new_callig"] = "row_pt"
    d["new_callig_pt"] = f"assets/fs6_row_{topic}.pt"
    d["data_csv"] = f"assets/fs6_{topic}_train.csv"
    d["in_mem_eval_sets"] = f"fewshot:assets/fs6_{topic}_eval.csv:{N_EVAL}"
    d["latent_shards_dir"] = f"data/50k/shards_fs6_{topic}"
    d["skel_latent_shards_dir"] = f"data/50k/shards_fs6_{topic}_std"
    d["eval_skel_latent_shards_dir"] = f"data/50k/shards_fs6_{topic}_std"
    d.update(FS_OVERRIDE)
    d["_comment"] = (f"v15 few-shot（{topic}）: v15a@150k 的 87-pair 表加第 88 行, "
                     f"冻结主干只训该行 (1536 参数)。号段 {ID_BASE + topic_slot(topic)*ID_SLOT_STRIDE}"
                     f".., latent 带 names 校验, w_repa=0, wd=0, use_ema=false。"
                     f"init=row_pt(该主题自己的 DINO K-Means 质心, 与建表同款构造)。")
    out = f"src/train/configs/v15_fs6_{topic}.json"
    json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


# ── 写 CSV ───────────────────────────────────────────────────────────────────
COLS = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
        "script_id", "character_id", "glyph_id", "aug", "std_path", "source",
        "src_image_path", "old_50k_id", "img_id"]


def write_csv(topic, cal, sc, sid, imgs, kind, id0, img_dir):
    idx = std_index()
    rows = []
    for n, ch in enumerate(sorted(imgs)):
        sp, k50, cid, gid = idx[(ch, sid)]
        rows.append({
            "image_path": os.path.join(img_dir, f"{ch}.png"),
            "calligrapher": cal, "script": sc, "character": ch,
            "calligrapher_id": NEW_RAW_ID, "script_id": sid,
            "character_id": cid, "glyph_id": gid, "aug": "",
            "std_path": sp, "source": "wild",
            "src_image_path": os.path.join(img_dir, f"{ch}.png"),
            # ⚠ old_50k_id 故意留空: extract_img_id 的回退链是 img_id -> old_50k_id,
            #   填了 50k 行号就等于给"号段隔离"埋一颗雷。溯源用 src_wild_path 列。
            "old_50k_id": "", "img_id": str(id0 + n),
        })
    out = f"assets/fs6_{topic}_{kind}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return out, rows


# ── 主流程 ───────────────────────────────────────────────────────────────────
def build(topic, dry_run=False):
    print(f"\n================ {topic} ================")
    cal, sc, sid, tr_map, ev_map = select(topic)
    slot = topic_slot(topic)
    id0 = ID_BASE + slot * ID_SLOT_STRIDE
    print(f"[选题] {cal}×{sc} (script_id={sid}) train={len(tr_map)} eval={len(ev_map)} "
          f"img_id {id0}..{id0 + len(tr_map) + len(ev_map) - 1}")
    if dry_run:
        return

    img_dir = f"data/50k/fs6_imgs_{topic}"
    stats = []
    for kind, mp in (("train", tr_map), ("eval", ev_map)):
        for ch, src in sorted(mp.items()):
            inv, gm, ink, w255 = fix_image(src, os.path.join(ROOT, img_dir, f"{ch}.png"))
            stats.append((kind, ch, inv, gm, ink, w255))
    n_inv = sum(1 for s in stats if s[2])
    ink = np.array([s[4] for s in stats])
    w255 = np.array([s[5] for s in stats])
    print(f"[图片] {len(stats)} 张 -> {img_dir}/  反相 {n_inv} 张  "
          f"二值化后 墨占比 med={np.median(ink):.3f} p90={np.percentile(ink, 90):.3f} | "
          f"纯白占比 med={np.median(w255):.3f} (50k=0.807/墨 0.191)")
    if np.median(ink) > 0.40:
        print(f"[图片] ⚠ 墨占比中位数 {np.median(ink):.3f} 明显高于 50k(0.19) —— "
              f"该主题笔画偏粗/底噪偏多, 判读时要记得这一项")

    _, tr_rows = write_csv(topic, cal, sc, sid, tr_map, "train", id0, img_dir)
    _, ev_rows = write_csv(topic, cal, sc, sid, ev_map, "eval", id0 + len(tr_map), img_dir)
    all_rows = tr_rows + ev_rows
    paths = [r["image_path"] for r in all_rows]
    ids = [int(r["img_id"]) for r in all_rows]
    names = [os.path.basename(r["image_path"]) for r in all_rows]
    stds = [r["std_path"] for r in all_rows]

    encode_shard(paths, ids, names, f"data/50k/shards_fs6_{topic}", "img")
    encode_shard(stds, ids, [os.path.basename(s) for s in stds],
                 f"data/50k/shards_fs6_{topic}_std", "std")

    dino_row([os.path.join(ROOT, r["image_path"]) for r in tr_rows],
             f"assets/fs6_row_{topic}.pt")
    map_path, new_pair = write_map(topic, sid)
    cfg = write_config(topic, new_pair, map_path)
    print(f"[配置] pair {NEW_RAW_ID}:{sid} -> {new_pair}  {map_path}\n       {cfg}")
    verify(topic)


# ── 自检（不通过就别去训） ────────────────────────────────────────────────────
def verify(topic):
    import glob
    tr = list(csv.DictReader(open(f"assets/fs6_{topic}_train.csv", encoding="utf-8")))
    ev = list(csv.DictReader(open(f"assets/fs6_{topic}_eval.csv", encoding="utf-8")))
    rows = tr + ev
    ids = [int(r["img_id"]) for r in rows]
    assert len(set(ids)) == len(ids), "img_id 重复"
    assert not ({r["character"] for r in tr} & {r["character"] for r in ev}), "train/eval 字重叠"
    # 1) 号段不得与 50k 的 std latent 重叠
    lib = set()
    for sp in glob.glob(f"{CK50K_STD}/shard_*.npz"):
        lib |= {int(x) for x in np.load(sp)["img_ids"]}
    hit = set(ids) & lib
    # 2) 两个 shard 的 names 必须逐行对上 CSV
    for tag, d in (("img", f"data/50k/shards_fs6_{topic}"),
                   ("std", f"data/50k/shards_fs6_{topic}_std")):
        f0 = sorted(glob.glob(f"{d}/shard_*.npz"))
        assert f0, f"缺 {d} 的 shard"
        got_ids, got_names = [], []
        for sp in f0:
            z = np.load(sp, allow_pickle=True)
            assert "names" in z, f"{sp} 没有 names -> 训练侧无法校验内容"
            got_ids += [int(x) for x in z["img_ids"]]
            got_names += [str(x) for x in z["names"]]
            n = z["latents"].shape
            assert n[1:] == (4, 32, 32), f"{sp} latent 形状 {n} 不是 (N,4,32,32)"
        want = ([os.path.basename(r["image_path"]) for r in rows] if tag == "img"
                else [os.path.basename(r["std_path"]) for r in rows])
        m = dict(zip(got_ids, got_names))
        bad = [(i, w, m[i]) for i, w in zip(ids, want) if m.get(i) != w]
        assert not bad, f"{tag} latent 内容与 CSV 不符 {len(bad)} 行: {bad[:3]}"
    # 3) 配置指向自己的 shard
    c = json.load(open(f"src/train/configs/v15_fs6_{topic}.json", encoding="utf-8"))
    for k in ("skel_latent_shards_dir", "eval_skel_latent_shards_dir", "latent_shards_dir"):
        assert f"fs6_{topic}" in c[k], f"{k}={c[k]} 没指向本主题"
    for k, v in FS_OVERRIDE.items():
        assert c[k] == v, f"配置 {k}={c[k]} != {v}"
    assert c["w_repa"] == 0.0 and c["weight_decay"] == 0.0
    # 4) latent 统计要对得上 50k（VAE/缩放口径不一致会一眼看出来）
    ref = np.load(sorted(glob.glob("data/50k/shards_img/shard_00000.npz"))[0])[
        "latents"][:512].astype(np.float32)
    mine = np.load(f"data/50k/shards_fs6_{topic}/shard_00000.npz")[
        "latents"].astype(np.float32)
    print(f"[自检] latent 口径  本主题 std={mine.std():.3f} mean={mine.mean():+.3f} | "
          f"50k std={ref.std():.3f} mean={ref.mean():+.3f}")
    assert abs(mine.std() - ref.std()) < 0.6, "latent 尺度与 50k 差太多，VAE 口径可能不对"
    print(f"[自检] ✓ {len(rows)} 行 | 号段与 {CK50K_STD} 重叠 {len(hit)} 个"
          f"{' (本主题 latent 自带 names, 已逐行校验过内容)' if not hit else ' ⚠'}"
          f" | img/std latent names 全对齐 | 配置 OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="+", help="形如 沈周-行")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    failed = []
    for t in a.topics:
        try:
            build(t.replace("_", "-"), dry_run=a.dry_run)
        except SystemExit as e:                    # 一个主题不合格不该带走整批
            print(f"[跳过] {t}: {e}")
            failed.append(t)
    if failed:
        print(f"\n未通过的主题: {failed}")
    if not a.dry_run:
        print("下一步: bash _sync_work/launch_fs6_stage1.sh")
