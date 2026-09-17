"""把 HCSU (wild / tie / bei) 的已二值化产物做成可直接训练的 256x256 数据 + CSV。

输入:  `/root/Workspace/xy/HCSU/_bin/{wild,tie,bei}/{书家}-{书体}/{字}.png`
       （由 tools/binarize_hcsu.py 产出; 原始数据只读保留）

三步:
  1. **几何归一化** —— 只对 wild 需要（实测尺寸 23~2427px 不统一）。
     紧裁切 bbox -> 最长边缩到 0.88*256=225 -> 居中贴 256x256 白底。
     tie/bei 已是 256x256, 直接复用。
     配方照 `tools/build_dataset_px60.py` 的 `SIZE=256, BOX_FRAC=0.88`。
  2. **渲染对应书体的 std 骨架** —— 照 `tools/build_std_skel_px60.py` 的 SCRIPT_FONT。
     ⚠ **渲染失败必须显式记录并剔除**, 绝不静默 fallback 到楷体
       (那会给草/篆目标喂楷形骨架, 书体不匹配, 且是静默错误)。
  3. **组装 CSV** —— 复用现有 glyph_id / calligrapher_id; 新字/新书家分配新 id。

书体范围: 只留 楷/行/隶（无草/篆字体）。
"""
import argparse, collections, csv, json, os, re, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None
ROOT = "/root/Workspace/xy/DiT"
HCSU = "/root/Workspace/xy/HCSU"
SIZE, BOX_FRAC = 256, 0.88
FONT_SIZE = 200                     # 照抄 build_std_skel_px60.py
ST = generate_binary_structure(2, 2)
FONT_DIR = f"{ROOT}/tools/fonts"
KEEP = {"楷", "行", "隶"}
SCRIPT_ID = {"楷": 0, "行": 3, "隶": 4}
SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}
# 书家名归一: HCSU 用异体写法, 与我们已有数据统一
NAME_ALIAS = {"文征明": "文徵明"}

_fc, _tofu = {}, {}


def get_font(fn, size=SIZE):
    k = (fn, size)
    if k not in _fc:
        p = os.path.join(FONT_DIR, fn)
        _fc[k] = ImageFont.truetype(p, size) if os.path.exists(p) else None
    return _fc[k]


def tofu_sig(fn, size=SIZE):
    if fn in _tofu:
        return _tofu[fn]
    f = get_font(fn, size)
    if f is None:
        _tofu[fn] = None
        return None
    img = Image.new("L", (size * 2, size * 2), 255)
    ImageDraw.Draw(img).text((size, size), "\ue000", font=f, fill=0, anchor="mm")
    a = np.asarray(img)
    m = a < 250
    _tofu[fn] = ("blank",) if not m.any() else (
        int(m.sum()), int(np.where(m.any(1))[0].ptp()), int(np.where(m.any(0))[0].ptp()))
    return _tofu[fn]


def render_std(ch, script, allow_fallback=False):
    """渲染标准字形**骨架**(白底黑细线, 3px), 返回 256x256 uint8 或 None。

    ⚠ 严格复刻 `tools/build_std_skel_px60.py:render_skel`:
        - FONT_SIZE=200, 直接在 256x256 上 anchor="mm" 居中,**不做 bbox 裁剪/缩放**
        - `skeletonize(a < 127)` -> `binary_dilation(ST, iterations=1)` -> 二值输出
      我们的 `std_path` 是**骨架**(墨点率 med 0.038)而不是填充字形(0.20),
      自己发明一套渲染方式会得到分布不同的 g 条件 —— 必须照抄。

    `allow_fallback=False`（默认）: 只用该书的字体。若该字在该书体下渲染不出,
      返回 None -> 上层**显式剔除**。避免"给行书目标喂楷形骨架"这种静默书体不匹配。
    `allow_fallback=True`: 额外尝试 simkai/simhei（照抄现有实现的行为, 覆盖更全但混书体）。
    """
    cands = list(SCRIPT_FONT.get(script, []))
    if allow_fallback:
        cands = cands + ["simkai.ttf", "simhei.ttf"]
    seen = set()
    for fn in cands:
        if fn in seen:
            continue
        seen.add(fn)
        fp = os.path.join(FONT_DIR, fn)
        if not os.path.isfile(fp):
            continue
        if fn not in _fc:
            try:
                _fc[fn] = ImageFont.truetype(fp, FONT_SIZE)
            except Exception:
                _fc[fn] = None
        font = _fc[fn]
        if font is None:
            continue
        img = Image.new("L", (SIZE, SIZE), 255)
        try:
            ImageDraw.Draw(img).text((SIZE // 2, SIZE // 2), ch, font=font, fill=0, anchor="mm")
        except Exception:
            continue
        a = np.asarray(img)
        if (a < 250).sum() < 10:              # 空白/豆腐
            continue
        sk = skeletonize(a < 127)
        if not sk.any():
            continue
        sk = binary_dilation(sk, ST, iterations=1)
        return np.where(sk, 0, 255).astype(np.uint8)
    return None


def normalize_img(src, dst):
    """紧裁切 bbox -> 最长边缩到 BOX_FRAC*SIZE -> 居中贴白底 256x256。"""
    a = np.asarray(Image.open(src).convert("L"))
    m = a < 128
    if not m.any():
        return False
    ys, xs = np.where(m)
    crop = Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    t = BOX_FRAC * SIZE
    s = t / max(crop.size)
    tw, th = max(int(crop.size[0] * s), 1), max(int(crop.size[1] * s), 1)
    crop = crop.resize((tw, th), Image.LANCZOS)
    cv = Image.new("L", (SIZE, SIZE), 255)
    cv.paste(crop, ((SIZE - tw) // 2, (SIZE - th) // 2))
    # ⚠ LANCZOS 缩放会引入抗锯齿灰阶。我们的数据是**纯二值**(实测 100% 2 阶),
    #   所以最后必须再二值化一次, 否则输入分布与训练数据不一致。
    out = np.where(np.asarray(cv) < 128, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(out, mode="L").save(dst, optimize=True)
    return True


def scan(root):
    out = []
    for dn in sorted(os.listdir(root)):
        dp = os.path.join(root, dn)
        if not os.path.isdir(dp):
            continue
        c, _, s = dn.rpartition("-")
        if s not in KEEP:
            continue
        for fn in sorted(os.listdir(dp)):
            if fn.lower().endswith(".png"):
                out.append((dn, c, s, os.path.splitext(fn)[0], os.path.join(dp, fn)))
    return out


def _norm_task(t):
    src, dst = t
    try:
        return normalize_img(src, dst)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="wild,tie,bei")
    ap.add_argument("--out", default=f"{ROOT}/data/hcsu_kxl")
    ap.add_argument("--csv", default=f"{ROOT}/assets/train_hcsu_kxl.csv")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude-callig", default="",
                    help="逗号分隔的书家名, 直接剔除(如 '沈周,傅山')。"
                         "傅山虽在名单里, 但可渲染后只剩 6 张, 样本量太小应剔除。")
    ap.add_argument("--allow-font-fallback", action="store_true",
                    help="字体渲染失败时额外试 simkai/simhei(书体可能不匹配)。"
                         "默认关闭 -> 显式剔除, 宁可少要数据也不要混书体的骨架。")
    args = ap.parse_args()
    args.exclude_callig = {x.strip() for x in args.exclude_callig.split(",") if x.strip()}

    out_img = f"{args.out}/imgs"
    out_std = f"{args.out}/std"

    # ---- 1. 收集 + 几何归一化 ----
    entries = []          # (tag, dn, callig, script, char, img_path)
    norm_tasks = []
    for tag in args.sources.split(","):
        tag = tag.strip()
        src_root = f"{HCSU}/_bin/{tag}"
        if not os.path.isdir(src_root):
            print(f"[skip] {tag}: {src_root} 不存在"); continue
        items = scan(src_root)
        print(f"[{tag}] 二值化产物 {len(items)} 张 (书体 {sorted(KEEP)})")
        for dn, c, s, ch, src in items:
            dst = os.path.join(out_img, tag, dn, f"{ch}.png")
            entries.append((tag, dn, c, s, ch, dst))
            if tag == "wild":              # 只有 wild 需要几何归一化
                norm_tasks.append((src, dst))
            else:
                norm_tasks.append((None, dst))   # 占位, 下面单独复制

    if args.dry_run:
        print(f"dry-run: {len(entries)} 条; wild 需归一化 "
              f"{sum(1 for e in entries if e[0]=='wild')} 张")
        return

    # wild 归一化 + tie/bei 直接复制
    real = [(f"{HCSU}/_bin/{t}/{dn}/{ch}.png", dst) for t, dn, c, s, ch, dst in entries]
    print(f"归一化/复制 {len(real)} 张 ...")
    ok = 0
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_norm_task, real, chunksize=64)):
            ok += bool(r)
            if (i + 1) % 5000 == 0:
                print(f"   ...{i+1}/{len(real)}  ok={ok}")
    print(f"   完成 ok={ok}/{len(real)}")

    # ---- 2. 渲染 std 骨架 ----
    keys = sorted({(e[3], e[4]) for e in entries})
    print(f"需渲染 (书体,字) 组合: {len(keys)}")
    std_cache, failed = {}, []
    for i, (s, ch) in enumerate(keys):
        arr = render_std(ch, s, allow_fallback=args.allow_font_fallback)
        if arr is None:
            failed.append((s, ch))
        else:
            std_cache[(s, ch)] = arr
        if (i + 1) % 2000 == 0:
            print(f"   ...{i+1}/{len(keys)}  失败累计 {len(failed)}")
    print(f"   渲染成功 {len(std_cache)}  失败 {len(failed)} "
          f"({100*len(failed)/max(1,len(keys)):.1f}%)")
    os.makedirs(out_std, exist_ok=True)
    for (s, ch), arr in std_cache.items():
        p = os.path.join(out_std, f"{s}_{ord(ch):05X}.png")
        Image.fromarray(arr, mode="L").save(p, optimize=True)

    # ---- 3. 组装 CSV ----
    old = list(csv.DictReader(open(f"{ROOT}/assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
    cid = {r["calligrapher"]: int(r["calligrapher_id"]) for r in old}
    gid = {r["character"]: int(r["glyph_id"]) for r in old}
    # ⚠ 新 calligrapher_id 必须从**词表已有键的最大值**之后开始, 不能从
    #   `max(old csv id)+1` 开始 —— base 词表里 9000~9010 已预留给别的书家,
    #   用 9002 起会**撞车**, 新书家静默拿到别人的 embedding 槽位(错标)。
    base_map_path = f"{ROOT}/assets/callig_id_map_base.json"
    base_map = json.load(open(base_map_path, encoding="utf-8"))
    _max_key = max(int(k) for k in base_map["id_map"])
    next_c = max(_max_key, max(cid.values())) + 1
    next_g = max(gid.values()) + 1
    print(f"新 calligrapher_id 从 {next_c} 开始 (词表最大键 {_max_key})")

    rows, skipped = [], []
    new_c, new_g = set(), set()
    for tag, dn, c, s, ch, dst in entries:
        cn = NAME_ALIAS.get(c, c)
        if cn in args.exclude_callig:
            skipped.append((tag, cn, s, ch)); continue
        if (s, ch) not in std_cache:
            skipped.append((tag, cn, s, ch)); continue
        if not os.path.exists(dst):
            skipped.append((tag, cn, s, ch)); continue
        if cn not in cid:
            cid[cn] = next_c; next_c += 1; new_c.add(cn)
        if ch not in gid:
            gid[ch] = next_g; next_g += 1; new_g.add(ch)
        std_p = os.path.join(out_std, f"{s}_{ord(ch):05X}.png")
        rows.append(dict(
            image_path=os.path.relpath(dst, ROOT), calligrapher=cn, script=s,
            character=ch, calligrapher_id=cid[cn], script_id=SCRIPT_ID[s],
            character_id=gid[ch], glyph_id=gid[ch], aug="", 
            std_path=os.path.relpath(std_p, ROOT), source=f"hcsu_{tag}"))

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 扩展 callig map —— **只从最终 rows 里出现的书家构造**, 避免剔除掉的书家留下孤儿槽位
    m = dict(base_map["id_map"])
    slot = max(m.values()) + 1
    added = {}
    present = {r["calligrapher"] for r in rows}
    for name in sorted(present):
        i = cid[name]
        if str(i) in m:
            continue                      # 已有书家(或其 id 已在词表), 沿用
        m[str(i)] = slot; added[name] = slot; slot += 1
    # 孤儿检查: rows 里每个书家的 id 都必须在词表里
    missing = [r["calligrapher"] for r in rows if str(r["calligrapher_id"]) not in m]
    if missing:
        raise RuntimeError(f"这些书家的 id 不在词表里(会静默落到 null): {sorted(set(missing))}")
    newmap = f"{ROOT}/assets/callig_id_map_hcsu.json"
    json.dump({"num_calligraphers": len(m), "id_map": m},
              open(newmap, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    rep = dict(rows=len(rows), skipped=len(skipped),
               new_calligraphers=sorted(new_c), n_new_calligrapher=len(new_c),
               n_new_glyph=len(new_g), std_render_fail=len(failed),
               callig_map=newmap, new_num_calligraphers=len(m),
               by_source=dict(collections.Counter(r["source"] for r in rows)))
    json.dump(rep, open(f"{args.out}/_build_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n=== 构建完成 ===")
    print(json.dumps({k: v for k, v in rep.items() if k != "new_calligraphers"},
                     ensure_ascii=False, indent=2))
    print(f"新增书家 {len(new_c)}: {sorted(new_c)}")
    print(f"剔除(无 std 骨架/无图) {len(skipped)}")
    print(f"CSV -> {args.csv}")
    print(f"callig map -> {newmap} (num_calligraphers {base_map['num_calligraphers']} -> {len(m)})")


if __name__ == "__main__":
    main()
