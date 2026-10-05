# -*- coding: utf-8 -*-
"""
in_mem_eval.py — 真·in-mem eval: 训练进程内, 暂停 stepping, 同卡采样+decode+指标
一次算完, 无 daemon、无 PNG 落盘、无 ckpt 重载。

用法 (config):
    "in_mem_eval": true,
    "in_mem_eval_sets": "seen:assets/eval_seen_v10.csv:10,strict:assets/eval_fame3_strict_clean_v9.csv:50",
    "eval_self_cond": true,          # noise400k 线的两遍自条件采样
    "eval_blend_alpha": 0.5,
    # 复用已有: eval_cfg, eval_steps, gpu_eval_img_root, gpu_eval_every, skel_latent_shards_dir

流程 (train.py 在 ckpt 保存点同步调用):
    ema_model (常驻 GPU, eval 模式) → make_eval_cache (模块级缓存, 跨 step 复用)
    → sample_latents[_self_cond] (bf16, dit_batch=16) → VAE decode in-mem (bf16)
    → PNG 落盘 (eval_samples_ctrl/step{step}/{seen|strict}/, g{i}.png + gt{i}.png)
    → _ssim/_mse → 追加 eval_stdskel_summary.csv / eval_stdskel_batch.csv
    (与 tools/eval/batch_eval.py 完全同格式 → registry / 历史曲线无缝兼容)

显存: 训练 17G + eval 模型常驻激活 + VAE ~0.6G + dit16/vae16 峰值 ~2G ≈ 19.5G < 24G。
"""
import csv
import glob
import os
import re
import time

import numpy as np
import torch
from PIL import Image

from src.eval.inference import (build_diffusion, load_eval_vae, make_eval_cache,
                                sample_latents, sample_latents_self_cond,
                                maybe_add_white, _mse, _ssim)

# 模块级缓存: 跨 step 复用 (eval cache / VAE / diffusion / id_map)
_CACHES = {}
_VAE = None
_VAE_REF = None      # ★ 2026-10-05: 缓存键 = VAE 路径 (否则第二个 VAE 复用第一个)
_DIFF = None
_CMAP = None
_WHITE_LAT_CACHE = {}       # 白底 latent (aux_zero_white 时 decode 前加回)

# ── LPIPS ─────────────────────────────────────────────────────────────────
# [v12+] 此前 summary/batch CSV 里 lpips_mean / lpips 两列**声明了但从来没算过**
# (写的一直是空串)。ssim 对"大面积白底匹配"不敏感, 会把"字形没生成出来"的
# 墨团判成不低的分 (doc59 §0 实测: 4ch 与 12ch 线 ssim 接近但图像完全不同)。
# LPIPS 对结构与细节敏感, 正是用来区分"容量够不够"与"ssim 饱和"的工具
# (doc62 §3)。实现照搬 auto_eval_ctrl.py 的可用版本, **跑在 CPU** ——
# in-mem eval 与训练同进程同卡, 走 CPU 可避免任何显存争用。
_LPIPS_FN = None
_LPIPS_LOADED = False
# ★ 2026-09-17: 记录**为什么**不可用，并在 summary 里醒目打印。
#   原行为: 失败只 print 一行 `lpips 列将留空`，然后 summary 里该列就是空的 ——
#   在几百行日志里极易漏看，而 LPIPS 正是用来区分"容量够不够"与"ssim 饱和"的
#   关键指标（doc62 §3）。**静默缺失等于丢掉了本次实验的主要结论依据。**
_LPIPS_FAIL_REASON = None


def lpips_status():
    """返回 LPIPS 的可用性描述（供 summary 打印）。"""
    if _LPIPS_FN is not None:
        return "ok"
    return f"UNAVAILABLE ({_LPIPS_FAIL_REASON})" if _LPIPS_FAIL_REASON else "not-requested"


def _get_lpips():
    """惰性加载 LPIPS(vgg) 到 CPU。不可用返回 None，但**原因会被记录并醒目上报**。"""
    global _LPIPS_FN, _LPIPS_LOADED, _LPIPS_FAIL_REASON
    if _LPIPS_LOADED:
        return _LPIPS_FN
    _LPIPS_LOADED = True
    try:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net="vgg", verbose=False)
        _LPIPS_FN.eval()
        for p in _LPIPS_FN.parameters():
            p.requires_grad_(False)
        print("[in-mem-eval] LPIPS(vgg) loaded on CPU")
    except Exception as _e:                                    # noqa: BLE001
        import traceback
        _LPIPS_FAIL_REASON = f"{type(_e).__name__}: {_e}"
        _LPIPS_FN = None
        print("=" * 70)
        print(f"[in-mem-eval] ⚠⚠ LPIPS 不可用 —— **lpips 列将全部为空**")
        print(f"    原因: {_LPIPS_FAIL_REASON}")
        print(f"    修法: pip install lpips  (或用 --in-mem-eval-lpips 0 显式关闭)")
        print(f"    影响: 本次实验将**无法判断容量是否饱和**(ssim 会饱和, LPIPS 不会)")
        print("=" * 70)
        traceback.print_exc()
    return _LPIPS_FN


def _lpips_per_sample(pred_np, gt_np, enabled=True):
    """pred_np/gt_np: (N,H,W,3) float32 [0,1] -> list[float] 或 None。

    转成 LPIPS 要求的 (N,3,H,W) [-1,1] 后一次批量 forward。
    """
    if not enabled:
        return None
    fn = _get_lpips()
    if fn is None:
        return None
    try:
        p = torch.from_numpy(np.ascontiguousarray(pred_np)).permute(0, 3, 1, 2).float()
        g = torch.from_numpy(np.ascontiguousarray(gt_np)).permute(0, 3, 1, 2).float()
        p = (p * 2.0 - 1.0).clamp(-1, 1)
        g = (g * 2.0 - 1.0).clamp(-1, 1)
        with torch.no_grad():
            d = fn(p, g)
        d = d.reshape(-1).cpu().numpy().astype(float)
        return [float(x) for x in d]
    except Exception as _e:                                    # noqa: BLE001
        print(f"[in-mem-eval] LPIPS forward failed ({_e!r}); lpips 列将留空")
        return None


def _get_vae(device, vae_path="data/pretrained/sd-vae-ft-ema"):
    """★ 2026-10-05: 按**路径**缓存。原来只判 `_VAE is None` -> 换 16ch flux VAE 时
    仍复用之前的 4ch sd-vae, 16ch latent 拿 4ch 解码器解 -> 指标静默变垃圾。"""
    global _VAE, _VAE_REF
    if _VAE is None or _VAE_REF != str(vae_path):
        _VAE = load_eval_vae(device, vae_path)
        _VAE_REF = str(vae_path)
    return _VAE


def _get_callig_map(path):
    """加载书家 id 映射表。

    ★ 2026-09-17: 原来 `if path and os.path.exists(path)` —— **路径不存在就静默返回 None**，
    调用方拿到 None 后不映射也不报错 -> 评测用**原始 id** 查书家表，
    与训练侧（用映射后的连续索引）**不是同一张表** -> 书家条件静默错位，
    而 ssim 照样算得出来（见 docs/system/70 §1.3）。

    现在: 配了路径但不存在 -> **直接抛错**；没配 -> 返回 None（合法，走原始 id）。
    """
    global _CMAP
    if _CMAP is not None:
        return _CMAP
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"callig_id_map 配了但不存在: {path!r}。\n"
            f"  若继续跑, 评测会用**原始书家 id** 而训练用的是映射后的连续索引 -> "
            f"书家条件静默错位。请修正路径, 或显式清空 callig_id_map 表示走原始 id。")
    from src.utils.callig_map import load_callig_id_map
    _CMAP, _ = load_callig_id_map(path)
    print(f"[in-mem-eval] callig_id_map loaded: {path}")
    return _CMAP


_CSMAP = None


def _get_callig_script_map(path):
    """(书家×书体) 联合风格词表加载(缓存单例)。没配 -> None(走单书家词表/原始 id)。

    与 _get_callig_map 同理: 配了路径但不存在 -> 直接报错(否则评测用单书家表
    而训练用 pair 表 -> 风格条件静默错位, ssim 照样算得出)。
    """
    global _CSMAP
    if _CSMAP is not None:
        return _CSMAP
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"callig_script_map 配了但不存在: {path!r}。若继续跑, 评测会用单书家词表 "
            f"而训练用的是 (书家,书体) pair -> 风格条件静默错位。")
    from src.utils.callig_script_map import load_callig_script_map
    _CSMAP = load_callig_script_map(path)
    print(f"[in-mem-eval] callig_script_map loaded: {path} "
          f"({_CSMAP['num_pairs']} pairs)")
    return _CSMAP


def _triple_remap_cache(cache, csv_path, n, args):
    """把 eval cache 的条件重映射到「三表」标签空间 (只在 cache 创建时调用一次)。

    v53: y_callig -> 书家表行 (原始 calligrapher_id 经 callig_remap);
         y_char   -> 汉字表行 (character_id 经 char_remap);
         y_script -> 书体表行 (script_id 经 font_remap)。
    ⚠ 必须幂等: cache 跨 step 复用, 若在命中路径也调用会二次重映射 (实测 KeyError)。
    """
    import json as _json

    def _load(p):
        if not p:
            return {}
        try:
            return {int(k): int(v) for k, v in
                    _json.load(open(p, encoding="utf-8")).items()}
        except Exception as _e:                                   # noqa: BLE001
            print(f"[in-mem-eval] ⚠ remap json 读取失败 {p}: {_e!r}")
            return {}

    _cr = _load(getattr(args, "callig_remap_json", ""))
    _hr = _load(getattr(args, "char_remap_json", ""))
    _fr = _load(getattr(args, "font_remap_json", ""))
    _rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))[:n]
    _miss = [k for k in ("callig", "char", "font") if not
             ({"callig": _cr, "char": _hr, "font": _fr}[k])]
    if _miss:
        raise RuntimeError(f"[in-mem-eval] use_script_cond 需要三份 remap json, 缺: {_miss}")
    cache["conds"] = [(_cr[int(r["calligrapher_id"])], _hr[int(r["character_id"])])
                      for r in _rows]
    _h = cache["hier_conds"]
    cache["hier_conds"] = (_h[0], _h[1], [_fr[int(s)] for s in _h[2]])
    cache["triple_mapped"] = True
    # ★ 2026-10-05 措辞修正: 这里映射的是三张**表**的 id (callig/char/font),
    #   与骨架(skel)无关; 旧文案 "v53 triple conds" 容易被误读成"给了骨架条件"。
    print(f"[in-mem-eval] cond tables remapped: callig/char/font ids "
          f"(n={len(_rows)}); 与骨架条件无关")


def _get_cache(csv_path, n, img_root, shards, args):
    """eval cache 跨 step 复用 (同一 set 每 2500 步重算一次无意义)。

    三表 (v53) 的重映射**只在本函数创建 cache 时**做一次:
    cache 命中时直接返回, 保证幂等 (重复重映射会 KeyError, 实测踩过)。
    """
    ck = (csv_path, n, shards)
    if ck not in _CACHES:
        sf = float(getattr(args, "vae_scaling_factor", 0.18215))
        cache = make_eval_cache(
            csv_path, img_root, None, 256, n, 8,
            int(getattr(args, "latent_channels", 4)), sf,
            skel_latent_shards_dir=shards,
            callig_id_map=_get_callig_map(getattr(args, "callig_id_map", None)),
            callig_script_map=_get_callig_script_map(getattr(args, "callig_script_map", None)))
        if getattr(args, "use_script_cond", False):
            _triple_remap_cache(cache, csv_path, n, args)
        _CACHES[ck] = cache
    return _CACHES[ck]


def _poster_canny(img):
    """从 gen 图现算 canny 边缘列。

    ⚠ 极性: 统一成**白底黑线**, 与数据侧约定一致。
    库原生输出 (cv2.Canny / 梯度阈值) 是"前景=255/背景=0"(黑底白线), 直接落盘
    会与 final_canny_base / skel3 PNG 的极性相反 -> 看起来像"数据错了"。
    这与 2026-09-14 的 E1(canny 黑底未归一)属同一类疏漏: 别把库的原生输出直接当交付格式。
    """
    import cv2
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, ky, borderType=cv2.BORDER_REFLECT)
    edge = np.sqrt(gx ** 2 + gy ** 2) > 150
    return Image.fromarray(np.where(edge, 0, 255).astype(np.uint8)).convert("RGB")


def _poster_skeleton(img):
    """从 gen 图现算骨架列。

    ⚠ 极性同 `_poster_canny`: 输出**白底黑线**(骨架=0), 而不是 skeletonize 的
    原生 "骨架=255" —— 否则与 skel3 PNG / 输入 g 的极性相反, 造成误读。
    """
    from skimage.morphology import skeletonize
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    bin_b = (gray > 127).astype(np.uint8)
    if gray.mean() > 127:
        bin_b = 1 - bin_b
    sk = skeletonize(bin_b.astype(bool))
    return Image.fromarray(np.where(sk, 0, 255).astype(np.uint8)).convert("RGB")


def save_input_g(results_dir, set_name, skels_latent, vae, sf, batch=16):
    """标准字输入 (g 条件) 落盘: eval_samples_ctrl/{set}_input_g/g{i}.png (幂等)。

    ⚠ **必须分批 decode**。原实现是

        dec = vae.decode(skels_latent / sf).sample      # 一次全解

    strict 集有 250 张，一次性 decode 的中间激活要 7.81 GiB，而训练常驻 20.25 GiB
    -> `CUDA out of memory`（实测踩过；该异常被上层 try 吞掉，只打一行
    `input-g save failed`，**poster 里悄悄少一整行图**，不报错）。
    主路径（预测图 decode）一直是按 vae_batch 分批的，这里照做即可。
    """
    out_dir = os.path.join(results_dir, "eval_samples_ctrl", f"{set_name}_input_g")
    os.makedirs(out_dir, exist_ok=True)
    n = skels_latent.shape[0]
    if all(os.path.exists(os.path.join(out_dir, f"g{i}.png")) for i in range(n)):
        return out_dir
    from PIL import Image as _Img
    dev = next(vae.parameters()).device
    bs = max(1, int(batch))
    for i in range(0, n, bs):
        j = min(i + bs, n)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dec = vae.decode(skels_latent[i:j].to(dev) / sf).sample
        dec = ((dec.float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
        for k in range(j - i):
            _Img.fromarray((dec[k] * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"g{i + k}.png"))
        del dec
        torch.cuda.empty_cache()
    return out_dir


_SAMECHAR_BYCHAR = {}


def _samechar_nn_eval(pred_np, gt_np, ev_rows, train_csv, max_cand=12, nw=16):
    """同字最近邻指标 —— 直接回答"生成结果最像训练集里谁写的"。

    对每一列, 在该字的训练集候选里找 SSIM 最高的那张 GT:
      nn_ssim  : max SSIM (均值 over 列)            "最像谁"
      nn_mean  : 同字候选的 SSIM 均值                "随机挑一张的期望"
      tgt_spec : own_gt - nn_mean                   **目标特异性**
      cal_hit/cal_base/cal_enrich                   有同书家候选时命中同书家 / 基线 / 富集倍数

    ⚠ 两个必须一起看才不误读的点:
      1. nn_ssim - nn_mean 天然 >0 (max≥mean), 不能单独当判别力证据 -> 要看 tgt_spec
      2. "最近邻是同书家"的比例会被"有多少列**有**同书家候选"限制 -> 必须看条件命中率与基线
    """
    import concurrent.futures as _cf
    if not (train_csv and len(pred_np) and ev_rows):
        return None
    if train_csv not in _SAMECHAR_BYCHAR:
        import collections as _co
        _by = _co.defaultdict(list)
        try:
            with open(train_csv, encoding="utf-8") as _f:
                for _r in csv.DictReader(_f):
                    _by[str(_r.get("character", ""))].append(_r)
        except Exception as _e:
            print(f"[samechar-nn] 训练 csv 不可读 ({_e!r})，跳过")
            _SAMECHAR_BYCHAR[train_csv] = None
            return None
        _SAMECHAR_BYCHAR[train_csv] = _by
    by_char = _SAMECHAR_BYCHAR[train_csv]
    if by_char is None:
        return None

    def _one(i):
        if i >= len(ev_rows):
            return None
        r = ev_rows[i]
        ch = str(r.get("character", ""))
        cal = r.get("calligrapher")
        cs = by_char.get(ch, [])
        if not cs:
            return None
        if len(cs) > max_cand:
            _st = len(cs) / max_cand
            cs = [cs[int(k * _st)] for k in range(max_cand)]
        own = float(_ssim(pred_np[i], gt_np[i]))
        ss = []
        for c in cs:
            p = c.get("image_path", "")
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.join(os.getcwd(), p)
            if not os.path.exists(p):
                continue
            # ⚠ 模块级是 `from PIL import Image`（没有 _Img 别名，_Img 只在 render_poster 内定义）。
            #   之前这里写 _Img.open 抛 NameError, 又被 `except Exception: continue` **静默吞掉**
            #   -> 所有候选被跳过 -> 函数返回 None, 不报错。这就是本项目反复出现的失败模式。
            #   现在只吞 IO 错误, 其它异常照常抛出。
            try:
                with Image.open(p) as _f:
                    t = np.asarray(_f.convert("RGB"), dtype=np.float32) / 255.0
            except (OSError, ValueError):
                continue
            if t.shape != pred_np[i].shape:
                continue
            ss.append((float(_ssim(pred_np[i], t)), c.get("calligrapher")))
        if not ss:
            return None
        vals = [s for s, _ in ss]
        ncal = sum(1 for _, c2 in ss if c2 == cal)
        hit = bool(ncal) and (max([s for s, c2 in ss if c2 == cal]) >= max(vals) - 1e-12)
        return dict(own=own, nn=max(vals), mc=float(np.mean(vals)),
                    ncal=ncal, n=len(ss), hit=hit)

    with _cf.ThreadPoolExecutor(max_workers=max(1, min(nw, len(pred_np)))) as _ex:
        rs = [x for x in _ex.map(_one, range(len(pred_np))) if x]
    if not rs:
        return None
    av = [x for x in rs if x["ncal"] > 0]
    _hit = (sum(1 for x in av if x["hit"]) / len(av)) if av else float("nan")
    _base = (sum(x["ncal"] / max(x["n"], 1) for x in av) / len(av)) if av else float("nan")
    return dict(n_ok=len(rs),
                nn_ssim=float(np.mean([x["nn"] for x in rs])),
                nn_mean=float(np.mean([x["mc"] for x in rs])),
                tgt_spec=float(np.mean([x["own"] - x["mc"] for x in rs])),
                cal_hit=_hit, cal_base=_base,
                cal_enrich=(_hit / _base) if (av and _base and _base == _base and _base > 0)
                else float("nan"),
                n_cal=len(av))


def _steps_scan(results_dir, sub):
    base = os.path.join(results_dir, "eval_samples_ctrl")
    steps = []
    for d in sorted(glob.glob(os.path.join(base, "step*"))):
        n = 0
        while os.path.exists(os.path.join(d, sub, f"g{n}.png")):
            n += 1
        if n:
            steps.append((int(re.search(r"step(\d+)", os.path.basename(d)).group(1)),
                          os.path.join(d, sub), n))
    return steps


import time as _time  # noqa: E402  (poster 计时用)

_TRAIN_REF_CACHE = {}
_font_cache = {}


def _train_ref_rows(train_csv, eval_csv, n_ev):
    """为每一列找两行训练集参照: [同书家·异字, 同字·异书家]。

    同书家优先同书体（结体习惯更可比）；同字取不同书家（看结构有多少种写法）。
    返回 None 表示不可用（调用方就少画这两行，不影响原 poster）。
    """
    key = (train_csv, eval_csv, n_ev)
    if key in _TRAIN_REF_CACHE:
        return _TRAIN_REF_CACHE[key]

    def _rows(p, n=0):
        if not (p and os.path.exists(p)):
            return []
        with open(p, encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        return r[:n] if n else r

    ev = _rows(eval_csv, n_ev)
    tr = _rows(train_csv)
    if not ev or not tr:
        _TRAIN_REF_CACHE[key] = None
        return None
    by_cal, by_char = {}, {}
    for r in tr:
        cid, sid = str(r.get("calligrapher_id", "")), str(r.get("script_id", ""))
        by_cal.setdefault((cid, sid), []).append(r)
        by_cal.setdefault(("*", cid), []).append(r)
        by_char.setdefault(str(r.get("character", "")), []).append(r)

    def _img(r):
        p = (r or {}).get("image_path", "")
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        return p if os.path.exists(p) else None

    out = [[], []]
    for i, r in enumerate(ev):
        ch, cid, sid = (r.get("character"), r.get("calligrapher_id"),
                        r.get("script_id"))
        c1 = None
        for k in [(str(cid), str(sid)), ("*", str(cid))]:
            cand = [x for x in by_cal.get(k, [])
                    if str(x.get("character", "")) != str(ch)]
            if cand:
                c1 = cand[i % len(cand)]
                break
        cand2 = [x for x in by_char.get(str(ch), [])
                 if str(x.get("calligrapher_id", "")) != str(cid)]
        out[0].append(_img(c1))
        out[1].append(_img(cand2[i % len(cand2)]) if cand2 else None)
    _TRAIN_REF_CACHE[key] = out
    return out


_TRAIN_NN_CACHE = {}


def _train_nn_row(train_csv, eval_csv, n_ev, gen_dir, max_cand=12):
    """训练集里**同字**、与生成图 SSIM 最高的那张 GT。

    返回 (paths, ssims, matched) 三个长度 n_ev 的列表；不可用时 None。
      paths   : 命中那张 GT 的路径（没命中为 None）
      ssims   : 对应 SSIM（用主评测口径 metrics.ssim, 高斯窗 win=11）
      matched : 命中行的 (character, calligrapher) 便于核对
    同字候选太多时按等间隔抽 max_cand 张（保证可复现）。
    """
    key = (train_csv, eval_csv, n_ev, gen_dir, max_cand)
    if key in _TRAIN_NN_CACHE:
        return _TRAIN_NN_CACHE[key]
    if not (train_csv and eval_csv and gen_dir and os.path.isdir(gen_dir)):
        _TRAIN_NN_CACHE[key] = None
        return None

    def _rows(p, n=0):
        if not (p and os.path.exists(p)):
            return []
        with open(p, encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        return r[:n] if n else r

    ev, tr = _rows(eval_csv, n_ev), _rows(train_csv)
    if not ev or not tr:
        _TRAIN_NN_CACHE[key] = None
        return None

    import numpy as _np
    from PIL import Image as _I
    from src.eval.metrics import ssim as _ssim_fn

    by_char = {}
    for r in tr:
        by_char.setdefault(str(r.get("character", "")), []).append(r)

    def _abs(p):
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        return p if os.path.exists(p) else None

    def _one(i, r):
        """一列: 载生成图 + 同字候选 -> 取 SSIM 最高那张。"""
        gp = os.path.join(gen_dir, f"g{i}.png")
        if not os.path.exists(gp):
            return None, float("nan"), None
        with _I.open(gp) as _f:
            g = _np.asarray(_f.convert("RGB"), dtype=_np.float32) / 255.0
        cands = by_char.get(str(r.get("character", "")), [])
        if len(cands) > max_cand:
            _st = len(cands) / max_cand
            cands = [cands[int(k * _st)] for k in range(max_cand)]
        best, bs, bm = None, -2.0, None
        for c in cands:
            cp = _abs(c.get("image_path", ""))
            if not cp:
                continue
            with _I.open(cp) as _f:
                t = _np.asarray(_f.convert("RGB"), dtype=_np.float32) / 255.0
            if t.shape != g.shape:
                continue
            s = float(_ssim_fn(g, t))
            if s > bs:
                bs, best = s, cp
                bm = (c.get("character", ""), c.get("calligrapher", ""))
        return best, (bs if best else float("nan")), bm

    # ★ 多线程: 列与列独立, 且 SSIM 底层是 scipy 卷积(释放 GIL) -> 线程有真实加速。
    #   不用多进程: 本函数在**训练进程内**被调用, 进程里有 CUDA, fork 有风险。
    import concurrent.futures as _cf
    _nw = int(os.environ.get("DIT_POSTER_NN_WORKERS", "16"))
    _nw = max(1, min(_nw, len(ev)))
    _t0 = _time.time()
    if _nw > 1:
        with _cf.ThreadPoolExecutor(max_workers=_nw) as _ex:
            _res = list(_ex.map(lambda _ir: _one(*_ir), enumerate(ev)))
    else:
        _res = [_one(i, r) for i, r in enumerate(ev)]
    paths = [x[0] for x in _res]
    ssims = [x[1] for x in _res]
    matched = [x[2] for x in _res]
    print(f"[poster] 同字最近邻 {len(ev)} 列, {_nw} 线程, {_time.time() - _t0:.1f}s", flush=True)
    _TRAIN_NN_CACHE[key] = (paths, ssims, matched)
    return _TRAIN_NN_CACHE[key]


def render_poster(results_dir, set_name, out=None, cell=224, gap=6,
                  train_csv=None, eval_csv=None, n_eval=None):
    """自动 poster (每 set 两张):

    主图 posters/{set}_poster.png:
      第 1 行 = 输入标准字 (eval_samples_ctrl/{set}_input_g/g{i}.png)
      中间每行 = 一个 ckpt 的生成结果 (时间升序, 行标签带 ssim)
      最后 1 行 = GT (最新 step 的 gt{i}.png)
    结构图 posters/{set}_struct.png (另放, 不挤主图):
      每 ckpt 一行, 每样本 gen | canny | skel 三列。
    均为全量重画并覆盖 → 永远是所有 step 的最新版。纯 CPU, 秒级。
    """
    from PIL import Image as _Img, ImageDraw
    sub = "g" if set_name in ("seen", "g") else set_name
    steps = _steps_scan(results_dir, sub)
    if not steps:
        return None
    n_max = max(s[2] for s in steps)
    # ★ 2026-10-01 修复: 原实现 cell = 1280//n_max (20 列 -> 64px) 且用 LANCZOS 缩小,
    #   1~2px 的骨架细线被邻域平均掉 -> poster 整行看起来"全白"(分数是在原生 256
    #   上算的, 未受影响)。POSTER_CELL / POSTER_NMAX 允许出可判读的大图。
    _cap = int(os.environ.get("POSTER_NMAX", "0"))
    if _cap > 0:
        n_max = min(n_max, _cap)
    cell = int(os.environ.get("POSTER_CELL", "0")) or max(
        64, min(224, 1280 // max(n_max, 1)))
    input_dir = os.path.join(results_dir, "eval_samples_ctrl", f"{set_name}_input_g")
    has_input = os.path.isdir(input_dir) and bool(glob.glob(os.path.join(input_dir, "g*.png")))

    font = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, cell // 8))
    font_small = _font_cache.get(cell) or _load_font(
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(9, cell // 7))
    _font_cache[cell] = font_small
    font_big = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(28, cell // 3))
    label_h = max(40, cell // 4)
    hdr_h = 56

    def _cell(path, bg, placeholder_label="N/A"):
        if path and os.path.exists(path):
            return _Img.open(path).convert("RGB").resize((cell, cell), _Img.LANCZOS)
        img = _Img.new("RGB", (cell, cell), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([1, 1, cell - 2, cell - 2], outline=(55, 60, 70), width=1)
        d.line([(4, 4), (cell - 5, cell - 5)], fill=(45, 48, 55), width=1)
        d.line([(4, cell - 5), (cell - 5, 4)], fill=(45, 48, 55), width=1)
        lbl = placeholder_label or "N/A"
        try:
            bbox = font_small.getbbox(lbl)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.text(((cell - tw) // 2, (cell - th) // 2), lbl, font=font_small, fill=(140, 145, 155))
        except Exception:
            pass
        return img

    # ── 主图: input 行 + 每 step gen 行 + GT 行 ──────────────────────────
    _ref = _train_ref_rows(train_csv, eval_csv, n_eval) if train_csv else None
    # ★ 同字最近邻: 用**最新 step** 的生成图
    _nn = (_train_nn_row(train_csv, eval_csv, n_eval, steps[-1][1])
           if (train_csv and steps) else None)
    W = cell * n_max + gap * 2
    n_rows = (len(steps) + 1 + (1 if has_input else 0) + (2 if _ref else 0)
              + (1 if _nn else 0))
    H = hdr_h + gap + n_rows * (label_h + cell + gap) + gap + 30
    canvas = _Img.new("RGB", (W, H), (15, 17, 22))
    draw = ImageDraw.Draw(canvas)
    y = gap
    draw.rectangle([0, y, W, y + hdr_h], fill=(0, 0, 0))
    # ★ 2026-10-05: 无 skel/glyph 条件时 has_input=False, 表头不能写 "input g"
    _in_lbl = "input g" if has_input else "no skel/glyph cond"
    _t = f"{set_name} (n={n_max}) — {_in_lbl} / per-ckpt gen / GT"
    if _ref:
        _t += " / 训练集类似条件 GT"
    if _nn:
        _t += " / 同字最像"
    draw.text((gap, y + 12), _t, font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    if has_input:
        draw.text((gap, y + 8), "input (标准字 g)", font=font_big, fill=(120, 220, 255))
        y += label_h
        for i in range(n_max):
            canvas.paste(_cell(os.path.join(input_dir, f"g{i}.png"), (30, 40, 60)),
                         (gap + i * cell, y))
        y += cell + gap
    for step, d, n in steps:
        draw.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                  font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            canvas.paste(_cell(os.path.join(d, f"g{i}.png"), (40, 40, 40)), (gap + i * cell, y))
        y += cell + gap
    draw.text((gap, y + 8), "GT", font=font_big, fill=(255, 160, 160))
    y += label_h
    gt_dir = steps[-1][1]
    for i in range(n_max):
        canvas.paste(_cell(os.path.join(gt_dir, f"gt{i}.png"), (50, 50, 50)), (gap + i * cell, y))
    y += cell + gap
    # ★ 训练集「类似条件」参照两行 —— 用来区分"不会这个字的结构"还是"不会这个书家的笔法"
    if _ref:
        for _lbl, _paths in zip(("训练集·同书家异字", "训练集·同字异书家"), _ref):
            draw.text((gap, y + 8), _lbl, font=font_big, fill=(170, 255, 170))
            y += label_h
            for i in range(n_max):
                _pth = _paths[i] if i < len(_paths) else None
                canvas.paste(_cell(_pth, (25, 25, 25)), (gap + i * cell, y))
            y += cell + gap
    # ★ 训练集「同字最像」一行: 格子角上标出 SSIM
    if _nn:
        _pnn, _snn, _mnn = _nn
        _v = [s for s in _snn if s == s]
        _mean = (sum(_v) / len(_v)) if _v else float("nan")
        draw.text((gap, y + 8),
                  f"训练集·同字最像 (SSIM 均值={_mean:.4f}, n={len(_v)})",
                  font=font_big, fill=(255, 220, 120))
        y += label_h
        _tag_h = max(12, cell // 5)
        for i in range(n_max):
            _pth = _pnn[i] if i < len(_pnn) else None
            canvas.paste(_cell(_pth, (25, 25, 25)), (gap + i * cell, y))
            _s = _snn[i] if i < len(_snn) else float("nan")
            if _s == _s:
                _txt = f"{_s:.2f}"
                draw.rectangle([gap + i * cell, y, gap + i * cell + 6 * len(_txt), y + _tag_h],
                               fill=(0, 0, 0))
                draw.text((gap + i * cell + 2, y + 1), _txt, font=font_small,
                          fill=(255, 220, 120))
        y += cell + gap
        # 数值单独落 CSV（poster 上只是角标, 不方便抄）
        try:
            _cp = os.path.join(results_dir, "posters",
                               f"{set_name}_train_samechar_nn.csv")
            os.makedirs(os.path.dirname(_cp), exist_ok=True)
            _ev_rows = []
            if eval_csv and os.path.exists(eval_csv):
                with open(eval_csv, encoding="utf-8") as _f:
                    _ev_rows = list(csv.DictReader(_f))[:n_eval]
            with open(_cp, "w", newline="", encoding="utf-8") as _f:
                _w = csv.writer(_f)
                _w.writerow(["idx", "eval_char", "eval_calligrapher",
                             "nn_ssim", "nn_char", "nn_calligrapher", "nn_image_path"])
                for i in range(n_max):
                    _r = _ev_rows[i] if i < len(_ev_rows) else {}
                    _m = _mnn[i] if i < len(_mnn) else None
                    _w.writerow([i, _r.get("character", ""), _r.get("calligrapher", ""),
                                 (f"{_snn[i]:.5f}" if i < len(_snn) and _snn[i] == _snn[i] else ""),
                                 (_m[0] if _m else ""), (_m[1] if _m else ""),
                                 (_pnn[i] or "") if i < len(_pnn) else ""])
            print(f"[poster] 同字最像 SSIM 已落盘: {_cp}")
        except Exception as _ce:
            print(f"[poster] 同字最像 CSV 落盘失败: {_ce!r}")

    out = out or os.path.join(results_dir, "posters", f"{set_name}_poster.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)

    # ── 结构图: input 行 + 每 step 行 (gen | 模型预测的 aux) + GT 行 ────────
    # 展示**模型自己预测的**结构通道 (aux_*{i}.png, 由 run_in_mem_eval 落盘),
    # 而不是从 gen 图重算 —— 后者只是"生成图的边缘", 无法判断模型是否学对了 aux 目标。
    _aux_cols = []
    for f in sorted(glob.glob(os.path.join(steps[-1][1], "aux_*0.png"))):
        _b = os.path.basename(f)
        _aux_cols.append(_b[len("aux_"):-len("0.png")])
    ncol = 1 + (len(_aux_cols) if _aux_cols else 2)
    W2 = cell * n_max * ncol + gap * 2
    n_rows2 = len(steps) + (1 if has_input else 0) + 1
    H2 = hdr_h + gap + n_rows2 * (label_h + cell + gap) + gap + 30
    canvas2 = _Img.new("RGB", (W2, H2), (15, 17, 22))
    draw2 = ImageDraw.Draw(canvas2)
    y = gap
    draw2.rectangle([0, y, W2, y + hdr_h], fill=(0, 0, 0))
    draw2.text((gap, y + 12),
               f"{set_name} structure — "
               + ("input g / " if has_input else "no skel/glyph cond / ")
               + "per-ckpt gen|"
               + ("|".join(_aux_cols) if _aux_cols
                  else "canny|skel(recomputed from gen, 非输入)")
               + " / GT",
               font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    if has_input:
        draw2.text((gap, y + 8), "input (标准字 g)", font=font_big, fill=(120, 220, 255))
        y += label_h
        for i in range(n_max):
            canvas2.paste(_cell(os.path.join(input_dir, f"g{i}.png"), (30, 40, 60)),
                          (gap + i * ncol * cell, y))
        y += cell + gap
    for step, d, n in steps:
        draw2.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                   font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            x = gap + i * ncol * cell
            gen = _cell(os.path.join(d, f"g{i}.png"), (40, 40, 40))
            canvas2.paste(gen, (x, y))
            if _aux_cols:
                for k, an in enumerate(_aux_cols):
                    canvas2.paste(_cell(os.path.join(d, f"aux_{an}{i}.png"), (30, 30, 30)),
                                  (x + (k + 1) * cell, y))
            else:   # 兼容: 老 ckpt 没有 aux 落盘时退回"从 gen 现算"
                canvas2.paste(_poster_canny(gen), (x + cell, y))
                canvas2.paste(_poster_skeleton(gen), (x + 2 * cell, y))
        y += cell + gap
    # GT 行: gt | 目标结构参照 (skel -> 骨架, 其余 -> 边缘; 均从 GT 图现算, 白底黑线)
    draw2.text((gap, y + 8), "GT", font=font_big, fill=(255, 160, 160))
    y += label_h
    _gt_dir2 = steps[-1][1]
    for i in range(n_max):
        x = gap + i * ncol * cell
        gti = _cell(os.path.join(_gt_dir2, f"gt{i}.png"), (50, 50, 50))
        canvas2.paste(gti, (x, y))
        if _aux_cols:
            for k, an in enumerate(_aux_cols):
                ref = _poster_skeleton(gti) if "skel" in an else _poster_canny(gti)
                canvas2.paste(ref, (x + (k + 1) * cell, y))
        else:
            canvas2.paste(_poster_canny(gti), (x + cell, y))
            canvas2.paste(_poster_skeleton(gti), (x + 2 * cell, y))
    out2 = os.path.join(results_dir, "posters", f"{set_name}_struct.png")
    canvas2.save(out2)
    return out


# ⚠ 系统里**没有** /usr/share/fonts/truetype/ 也没有任何 CJK 字体，
#   原实现恒回退 ImageFont.load_default() -> poster 的中文标签全是方块/空白。
#   项目自带 _fonts/ 有中文字体，优先用它们。
_FONT_CANDIDATES = (
    "_fonts/msyh.ttc", "_fonts/Deng.ttf", "_fonts/STSONG.TTF",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _load_font(fp, size):
    from PIL import ImageFont
    for c in [fp] + [x for x in _FONT_CANDIDATES if x != fp]:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _step_ssim_txt(results_dir, step, set_name):
    """poster 标签用: 从 summary csv 读该 (step, set) 的 ssim, 缺失返回 '-'. """
    _sp = os.path.join(results_dir, "eval_stdskel_summary.csv")
    try:
        for _r in csv.DictReader(open(_sp, encoding="utf-8")):
            if int(_r["step"]) == int(step) and _r["set"] == set_name:
                return f"ssim={float(_r['ssim_mean']):.4f}"
    except Exception:                                            # noqa: BLE001
        pass
    return "-"


def run_in_mem_eval(model, args, step, device, results_dir=None, logger=print):
    """在训/独立评测核心: 按 `--in-mem-eval-sets` 逐集采样并评指标。

    model      : 已 eval() 的模型 (优先 EMA 权重; 不要传 torch.compile 包装)
    results_dir: 落盘根 (summary/batch csv, posters/, eval_samples_ctrl/)
    logger     : 可调用对象 (默认 print; train.py 传 logger.info)

    ★ 2026-10-05: 以下 ssim/ink_ssim/ink_iou/skel_iou 全是**事后度量** ——
      skel_iou 由「生成图/GT 图各自 skeletonize 后比 IoU」算出, 与是否给
      骨架条件无关 (无 skel 条件的实验同样有 skel_iou)。

    返回 {set_name: {ssim, mse, lpips, ink_ssim, ink_iou, skel_iou, frag, hole,
                     n, nn_ssim, nn_mean, tgt_spec, cal_enrich}}
    """
    global _DIFF
    import time
    results_dir = str(results_dir or getattr(args, "results_dir", "") or ".")
    os.makedirs(results_dir, exist_ok=True)

    # ── 采样参数 (全部来自 args, 缺省保守值) ──
    img_root = None                       # csv image_path 已含完整相对路径
    shards = (getattr(args, "eval_skel_latent_shards_dir", "") or ""
              or getattr(args, "skel_latent_shards_dir", "") or "")
    cfg_scale = float(getattr(args, "eval_cfg", 0.7))
    ddim_steps = int(getattr(args, "eval_steps", 50))
    dit_batch = int(getattr(args, "in_mem_eval_batch", 16))
    vae_batch = int(getattr(args, "in_mem_eval_vae_batch", 16))
    sf = float(getattr(args, "vae_scaling_factor", 0.18215))
    use_self_cond = bool(getattr(args, "eval_self_cond", False))
    blend_alpha = float(getattr(args, "eval_blend_alpha", 0.0))
    if _DIFF is None:
        _DIFF = build_diffusion(ddim_steps, str(getattr(args, "diffusion_type", "flow")))

    # ── 解析评测集: "name:csv:n,name:csv:n" ──
    sets = []
    for _spec in str(getattr(args, "in_mem_eval_sets", "") or "").split(","):
        _spec = _spec.strip()
        if not _spec:
            continue
        _p = _spec.split(":")
        sets.append((_p[0],
                     _p[1] if len(_p) > 1 and _p[1] else getattr(args, "eval_csv", ""),
                     int(_p[2]) if len(_p) > 2 and _p[2] else 0))
    if not sets:
        logger("[in-mem-eval] in_mem_eval_sets 为空 -> 跳过评测")
        return {}

    sum_path = os.path.join(results_dir, "eval_stdskel_summary.csv")
    raw_path = os.path.join(results_dir, "eval_stdskel_batch.csv")
    # ★★ 2026-10-03 修 (P0-1): 表头检查必须放在 open(..., "a") **之前**。
    #   旧顺序是 [open("a") -> exists() -> 读表头], 而 "a" 模式会**创建空文件**,
    #   于是 exists() 恒真、表头恒读到 0 列 -> 每次 eval 都误判"表头不匹配"。
    #   更致命的是 rename 时 f_sum 句柄仍指向旧 inode, writer 继续往**改名后的
    #   备份文件**里写 -> 磁盘上主 summary 永远不存在 (实测 runs_AB 只有 3 个
    #   .bak_oldcols 而无主文件, 每个 .bak 恰好含一次 eval 的数据)。
    _SUM_HDR = ["exp", "step", "set", "n", "ssim_mean", "ssim_p10", "ssim_q1",
                "ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean",
                "ink_ssim_mean", "ink_iou_mean", "skel_iou_mean",
                "frag_ratio", "hole_pred", "hole_gt",
                "nn_ssim", "nn_mean", "tgt_spec", "cal_enrich"]
    if os.path.exists(sum_path):
        try:
            with open(sum_path, encoding="utf-8") as _f:
                _got = next(csv.reader(_f), [])
        except StopIteration:
            _got = []
        if _got != _SUM_HDR:
            _bak = sum_path + ".bak_oldcols"
            if os.path.exists(_bak):
                _bak = sum_path + f".bak_oldcols.{int(os.path.getmtime(sum_path))}"
            os.rename(sum_path, _bak)
            print(f"[in-mem-eval] ⚠ summary 表头不匹配({len(_got)} vs {len(_SUM_HDR)} 列) "
                  f"-> 旧文件备份为 {os.path.basename(_bak)}, 重开新表")
    # 检查完再读已完成项 (此时文件要么不存在, 要么表头已验证)
    done = set()
    if os.path.exists(sum_path):
        for r in csv.DictReader(open(sum_path, encoding="utf-8")):
            done.add((int(r["step"]), r["set"]))
    new_sum = not os.path.exists(sum_path)
    new_raw = not os.path.exists(raw_path)
    # ★ 句柄只在表头确认之后才打开; 之后不会再 rename 本文件 -> 不会再产生孤儿写入。
    f_sum = open(sum_path, "a", newline="", encoding="utf-8")
    f_raw = open(raw_path, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    w_raw = csv.writer(f_raw)
    if new_sum:
        w_sum.writerow(_SUM_HDR)
        f_sum.flush()
    # ★ 累积自检: 每次 eval 打印"主文件行数 + 已完成 (step,set) 对数"。
    #   若曲线再被重置, 这行会立刻显示行数归零/不增长 -> 一眼可见, 不必事后考古。
    try:
        _nlines = sum(1 for _ in open(sum_path, encoding="utf-8")) if os.path.exists(sum_path) else 0
    except Exception:                                            # noqa: BLE001
        _nlines = -1
    print(f"[eval] summary {sum_path}: {_nlines} 行, 已完成 (step,set) {len(done)} 对, "
          f"表头 {len(_SUM_HDR)} 列{' (本次新建)' if new_sum else ''}", flush=True)

    if new_raw:
        w_raw.writerow(["exp", "step", "set", "idx", "img_id", "char", "script",
                        "calligrapher", "mse", "ssim", "lpips",
                        "ink_ssim", "ink_iou", "skel_iou",
                        "frag_ratio", "hole_pred", "hole_gt"])
    exp = os.path.basename(results_dir.rstrip("/"))
    out = {}

    try:
        for name, csvp, n in sets:
            if (step, name) in done:
                print(f"[in-mem-eval] ⚠ step={step} set={name} 已记录在 {sum_path} 里"
                      f" -> 跳过。同一 results_dir 按 (step,set) 去重，**换数据重跑**时"
                      f"整轮评测会这样静默跳过；重跑请换一个新的 results_dir。")
                continue
            # ★ 2026-09-28: seen 集用训练集的 skel_latent_shards_dir，strict 集用 eval_skel_latent_shards_dir
            # ★ 2026-09-29: **双口径 (GT / predskel)**。set 名带 `_pred` 后缀时改用 pred 骨架目录:
            #     seen_pred   -> skel_latent_shards_dir_pred      (SkelNet 的 predskel)
            #     strict_pred -> eval_skel_latent_shards_dir_pred
            #   未配置 pred 目录则**回退到 GT 目录**(行为与旧版逐位一致, 不会静默出错)。
            #   poster 在下面按 set 名渲染, 因此两套口径自动各出一张海报。
            _is_pred = name.endswith("_pred")
            _base = name[:-5] if _is_pred else name
            _seen_like = _base in ("seen", "train", "g")
            _skey = (("skel_latent_shards_dir" if _seen_like
                      else "eval_skel_latent_shards_dir")
                     + ("_pred" if _is_pred else ""))
            set_shards = getattr(args, _skey, "") or ""
            if not set_shards:          # 未配 pred -> 回退 GT
                set_shards = ((getattr(args, "skel_latent_shards_dir", "") or shards)
                              if _seen_like else
                              (getattr(args, "eval_skel_latent_shards_dir", "") or ""
                               or getattr(args, "skel_latent_shards_dir", "") or ""))
            if _is_pred:
                print(f"[in-mem-eval] set={name} (pred 口径) skel={set_shards}")
            cache = _get_cache(csvp, n, img_root, set_shards, args)
            n = cache["n"]
            # ★ 2026-09-18: 读源 csv 行，用于把 img_id/char/script/**calligrapher**
            #   写进逐样本 CSV。原来这几列全是空串，导致无法把"每个书家的 strict ssim"
            #   与"该书家的风格条件强度"对齐做散点（区分瓶颈在数据还是条件机制）。
            _src = []
            try:
                _src = list(csv.DictReader(open(csvp, encoding="utf-8")))[:n]
            except Exception:
                _src = []
            t0 = time.time()
            if use_self_cond:
                lat = sample_latents_self_cond(
                    model, _DIFF, cache["noise"], cache["conds"],
                    cfg_scale, dit_batch, device,
                    skel=cache["skels_latent"], seed=0, blend_alpha=blend_alpha,
                    hier_conds=cache.get("hier_conds"))
            else:
                lat = sample_latents(
                    model, _DIFF, cache["noise"], cache["conds"],
                    cfg_scale, dit_batch, device,
                    skel=cache["skels_latent"], seed=0,
                    hier_conds=cache.get("hier_conds"))
            t_s = time.time() - t0

            vae = _get_vae(device, str(getattr(args, "eval_vae_path", "") or "")
                           or "data/pretrained/sd-vae-ft-ema")
            gts = (cache["gts"].to(device) + 1) / 2
            preds = torch.empty_like(gts)
            _zw = bool(getattr(args, "aux_zero_white", False))
            # aux 组名 (顺序 == aux_latent_shards_dirs) —— 落盘 aux_{name}{i}.png,
            # struct poster 直接展示**模型预测的**结构通道 (而不是从 gen 图重算)。
            _aux_names = []
            for _d in (getattr(args, "aux_latent_shards_dirs", "") or "").split(","):
                _d = _d.strip()
                if not _d:
                    continue
                _tk = [t for t in os.path.basename(_d).split("_") if t]
                _aux_names.append(_tk[1] if len(_tk) > 1 else f"aux{len(_aux_names)}")
            _sub = "g" if name in ("seen", "g") else name
            _sd = os.path.join(results_dir, "eval_samples_ctrl", f"step{int(step):07d}", _sub)
            _save = bool(getattr(args, "in_mem_eval_save_samples", True))
            # ★ 2026-10-05 (16ch Flux): 图像 latent 的通道数**不能硬编码 4**。
            #   Flux AE 下图像就是全部 16 通道; 硬切成 4 通道会让 16ch VAE 解码报
            #   "expected input[...,4,32,32] to have 16 channels" (实测冒烟抓到)。
            #   12ch aux 配方 (latent_channels=4 + 4*len(aux)) 语义不变。
            _n_img = int(getattr(args, "latent_channels", 4) or 4)
            _n_aux = max(0, (lat.shape[1] - _n_img) // 4)
            for i in range(0, n, vae_batch):
                j = min(i + vae_batch, n)
                _lat = lat[i:j].to(device)
                _aux_lat = None
                if _lat.shape[1] > _n_img:
                    _aux_lat = _lat[:, _n_img:]
                    _lat = _lat[:, :_n_img]
                # 白底归零 (aux_zero_white): 统一走 maybe_add_white, 勿内联 (防漂移/漏改)
                _lat = maybe_add_white(_lat, _zw)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec = vae.decode(_lat / sf).sample
                preds[i:j] = (dec.clamp(-1, 1) + 1) / 2
                # aux 通道同样减过白底 -> 也要加回再 decode, 否则整列发黄/发黑
                if _aux_lat is not None and _save:
                    from PIL import Image as _AImg
                    _aux_lat = maybe_add_white(_aux_lat, _zw)
                    os.makedirs(_sd, exist_ok=True)
                    for _g in range(_n_aux):
                        _nm = _aux_names[_g] if _g < len(_aux_names) else f"aux{_g}"
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _da = vae.decode(_aux_lat[:, _g * 4:(_g + 1) * 4] / sf).sample
                        for _k in range(_da.shape[0]):
                            _ar = ((_da[_k].float().clamp(-1, 1) + 1) / 2
                                   ).cpu().numpy().transpose(1, 2, 0)
                            _AImg.fromarray((_ar * 255).astype(np.uint8)).save(
                                os.path.join(_sd, f"aux_{_nm}{i + _k}.png"))
            pred_np = preds.cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gts.cpu().numpy().transpose(0, 2, 3, 1)

            # PNG 落盘 (与 batch_eval --save-samples 同路径): g{i}.png + gt{i}.png
            if bool(getattr(args, "in_mem_eval_save_samples", True)):
                from PIL import Image as _Img
                _sub = "g" if name in ("seen", "g") else name
                _sd = os.path.join(results_dir, "eval_samples_ctrl", f"step{int(step):07d}", _sub)
                os.makedirs(_sd, exist_ok=True)
                for i in range(n):
                    _Img.fromarray((pred_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"g{i}.png"))
                    _Img.fromarray((gt_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"gt{i}.png"))
                # 标准字输入 (g 条件) 落盘 (poster 第 1 行, 跨 step 复用, 幂等)
                if cache.get("skels_latent") is not None:
                    try:
                        save_input_g(results_dir, name, cache["skels_latent"], vae, sf,
                                     batch=vae_batch)
                    except Exception as _se:
                        # ★ 2026-09-18: 加 traceback。原来只打一行 repr，
                        #   OOM 被吞掉后只看到 "input-g save failed: OutOfMemoryError(...)"，
                        #   不知道是哪一步、也不知道 poster 会少一整行图。
                        import traceback as _tb
                        logger(f"[in-mem-eval] ✗ input-g save failed "
                               f"({type(_se).__name__}): {_se}\n"
                               f"{_tb.format_exc()}\n"
                               f"  -> poster 第 1 行（标准字输入）会缺失，但**指标不受影响**")

            ssims, mses = [], []
            # ★ 墨迹域指标：全图 SSIM 被 ~90% 白底严重抬高（实测 0.5745 vs 墨迹框 0.3161）。
            #   对齐 tools/batch_eval 的口径：ink_ssim / ink_iou / skel_iou 才是
            #   "字写得对不对" 的真实反映。
            _inks, _inkious, _skels, _frags = [], [], [], []
            _holes_p, _holes_g = [], []
            try:
                from src.eval.metrics_ink import ink_ssim as _ink_ssim, ink_iou as _ink_iou
                from src.eval.metrics import (skel_iou as _skel_iou,
                                              frag_ratio as _frag_ratio,
                                              hole_ratio as _hole_ratio)
                _has_ink = True
            except Exception as _e:
                logger(f"[in-mem-eval] ⚠ 墨迹指标不可用 ({_e!r})，ink 列将留空")
                _has_ink = False
            for i in range(n):
                mses.append(_mse(pred_np[i], gt_np[i]))
                ssims.append(_ssim(pred_np[i], gt_np[i]))
                if _has_ink:
                    _inks.append(_ink_ssim(pred_np[i], gt_np[i]))
                    _inkious.append(_ink_iou(pred_np[i], gt_np[i]))
                    _skels.append(_skel_iou(pred_np[i], gt_np[i], thresh=0.5))
                    _frags.append(_frag_ratio(pred_np[i], gt_np[i], thresh=0.5))
                    _holes_p.append(_hole_ratio(pred_np[i], thresh=0.5))
                    _holes_g.append(_hole_ratio(gt_np[i], thresh=0.5))
            ssim = np.array(ssims)
            mse = float(np.mean(mses))
            _ink_ssim_mean = f"{float(np.mean(_inks)):.5f}" if _inks else ""
            _ink_iou_mean = f"{float(np.mean(_inkious)):.5f}" if _inkious else ""
            _skel_iou_mean = f"{float(np.mean(_skels)):.5f}" if _skels else ""
            _frag_mean = f"{float(np.mean(_frags)):.4f}" if _frags else ""
            _hole_p_mean = f"{float(np.mean(_holes_p)):.4f}" if _holes_p else ""
            _hole_g_mean = f"{float(np.mean(_holes_g)):.4f}" if _holes_g else ""
            _ink_txt = (f" ink_ssim={float(np.mean(_inks)):.4f}" if _inks else "")
            _frag_txt = (f" frag={float(np.mean(_frags)):.3f}" if _frags else "")
            _hole_txt = (f" hole={float(np.mean(_holes_p)):.3f}"
                         f"/{float(np.mean(_holes_g)):.3f}" if _holes_p else "")
            # LPIPS (v12+): 默认开启, 可用 --in-mem-eval-lpips 0 关闭。
            # ssim 会被大面积白底匹配骗过 (doc59), LPIPS 对结构细节敏感。
            _lp = _lpips_per_sample(
                pred_np, gt_np,
                enabled=bool(getattr(args, "in_mem_eval_lpips", True)))
            _lp_mean = f"{float(np.mean(_lp)):.5f}" if _lp else ""
            _lp_txt = f" lpips={float(np.mean(_lp)):.4f}" if _lp else " lpips=NA"
            q10, q25, q50, q75, q90 = np.percentile(ssim, [10, 25, 50, 75, 90])
            # ★ 同字最近邻指标（"生成结果最像训练集里谁写的"）
            _scn = None
            if getattr(args, "in_mem_eval_samechar_nn", True):
                try:
                    _scn = _samechar_nn_eval(
                        pred_np, gt_np, _src, getattr(args, "data_csv", None))
                except Exception as _sce:
                    logger(f"[in-mem-eval] ⚠ 同字最近邻计算失败: {_sce!r}")
            _scn_txt = ""
            if _scn:
                _scn_txt = (f" | nn={_scn['nn_ssim']:.4f} tgt_spec={_scn['tgt_spec']:+.4f}"
                            f" cal_enrich={_scn['cal_enrich']:.2f}x")
            w_sum.writerow([exp, step, name, n, f"{ssim.mean():.4f}",
                            f"{q10:.4f}", f"{q25:.4f}", f"{q50:.4f}",
                            f"{q75:.4f}", f"{q90:.4f}", f"{mse:.5f}", _lp_mean,
                            _ink_ssim_mean, _ink_iou_mean, _skel_iou_mean, _frag_mean,
                            _hole_p_mean, _hole_g_mean,
                            (f"{_scn['nn_ssim']:.4f}" if _scn else ""),
                            (f"{_scn['nn_mean']:.4f}" if _scn else ""),
                            (f"{_scn['tgt_spec']:+.4f}" if _scn else ""),
                            (f"{_scn['cal_enrich']:.3f}" if _scn else "")])
            # ★ 2026-10-03: 必须显式 flush。原来只靠进程退出时刷新 -> 下一次 eval 打开
            #   本文件读表头时读到的是**空文件**, 判定表头不匹配 -> 把 summary 改名备份
            #   (_bak_oldcols.*), 于是训练曲线每次 eval 被重置 (实测一晚上产生 10 个备份,
            #   曲线全散在备份里)。flush 后曲线连续累积。
            f_sum.flush()
            f_raw.flush()
            for i in range(n):
                _s = _src[i] if i < len(_src) else {}
                w_raw.writerow([exp, step, name, i,
                                _s.get("image_path", ""),
                                _s.get("character", ""),
                                _s.get("script", ""),
                                _s.get("calligrapher", ""),
                                f"{mses[i]:.5f}", f"{ssims[i]:.4f}",
                                (f"{_lp[i]:.5f}" if _lp else ""),
                                (f"{_inks[i]:.5f}" if _inks else ""),
                                (f"{_inkious[i]:.5f}" if _inkious else ""),
                                (f"{_skels[i]:.5f}" if _skels else ""),
                                (f"{_frags[i]:.4f}" if _frags else ""),
                                (f"{_holes_p[i]:.4f}" if _holes_p else ""),
                                (f"{_holes_g[i]:.4f}" if _holes_g else "")])
            f_sum.flush()
            f_raw.flush()
            # ★ 2026-10-03: 补齐数值字段 —— 早停 (iou_lpips) 需要 lpips / skel_iou,
            #   而原来只回传 4 个键 -> 早停拿不到判据。
            out[name] = {"ssim": float(ssim.mean()),
                         "mse": float(mse),
                         "lpips": float(np.mean(_lp)) if _lp else None,
                         "ink_ssim": float(np.mean(_inks)) if _inks else None,
                         "ink_iou": float(np.mean(_inkious)) if _inkious else None,
                         "skel_iou": float(np.mean(_skels)) if _skels else None,
                         "frag": float(np.mean(_frags)) if _frags else None,
                         "hole": float(np.mean(_holes_p)) if _holes_p else None,
                         "hole_gt": float(np.mean(_holes_g)) if _holes_g else None,
                         "n": int(n),
                         "nn_ssim": (float(_scn["nn_ssim"]) if _scn else None),
                         "nn_mean": (float(_scn["nn_mean"]) if _scn else None),
                         "tgt_spec": (float(_scn["tgt_spec"]) if _scn else None),
                         "cal_enrich": (float(_scn["cal_enrich"]) if _scn else None)}
            logger(f"[in-mem-eval] step={step} set={name} n={n} "
                   f"ssim={ssim.mean():.4f} (med={q50:.4f}) mse={mse:.5f}"
                   f"{_lp_txt}{_ink_txt}{_frag_txt}{_hole_txt}{_scn_txt} "
                   f"sample={t_s:.0f}s total={time.time()-t0:.0f}s")
            # 自动 poster: 全量重画该 set 所有 step (秒级, 覆盖旧文件)
            try:
                _p = render_poster(results_dir, name,
                                   train_csv=getattr(args, "data_csv", None),
                                   eval_csv=csvp, n_eval=n)
                if _p:
                    logger(f"[in-mem-eval] poster updated: {_p}")
            except Exception as _pe:
                logger(f"[in-mem-eval] poster render failed: {_pe!r}")
    finally:
        f_sum.close()
        f_raw.close()
        torch.cuda.empty_cache()
    return out


def _fmt_eval(name, v):
    """汇总行。缺指标时打印 NA, 不让格式化把整次评测打成失败。"""
    if not isinstance(v, dict):
        return f"{name} ssim={v:.4f}"

    def n(x, spec):
        return "NA" if x is None else format(x, spec)

    return (f"{name} ssim={n(v.get('ssim'), '.4f')} "
            f"ink={n(v.get('ink_ssim'), '.4f')} "
            f"frag={n(v.get('frag'), '.3f')} "
            f"hole={n(v.get('hole'), '.3f')}")


def maybe_run_in_training(args, ema_model, model, train_steps, device,
                          checkpoint_dir, logger, is_eval_step, inline=True):
    """训练循环里的在训评测入口 —— 把"是否该跑 / 用哪份权重 / 计时 / 报错"都收在这里。

    train.py 侧只需一行调用：

        from src.eval.in_mem_eval import maybe_run_in_training
        maybe_run_in_training(args, ema_model, model, train_steps, device,
                              checkpoint_dir, logger, is_eval_step=_save_ckpt,
                              inline=_EVAL_INLINE)

    行为（与原内联实现完全一致）：
      * 只有 `inline and --in-mem-eval and is_eval_step` 才跑
      * 优先用常驻 EMA 权重（与 ckpt 落盘的是同一份），没有 EMA 就用裸模型
      * 显存：训练 ~17G + eval ~2G ≈ 19.5G < 24G
      * **失败只 warning 不中断训练**（评测不该把训练带崩），但会打完整 traceback
    返回 results dict，未跑则返回 None。
    """
    if not (inline and getattr(args, "in_mem_eval", False) and is_eval_step):
        return None
    import time
    _em = (ema_model if ema_model is not None
           else (model.module if hasattr(model, "module") else model))
    _em.eval()
    t0 = time.time()
    try:
        res = run_in_mem_eval(
            _em, args, train_steps, device,
            # ★ 2026-10-04: 评测产物必须落在**本次 run 自己的**目录 (train.py 已把
            #   args.eval_dir 解析为 run 目录), 否则共用目录下 (step,set) 去重会
            #   把整轮评测静默跳过。兜底: 由 checkpoint_dir 反推 run 目录。
            results_dir=str(getattr(args, "eval_dir", "") or "").strip()
            or os.path.dirname(os.path.dirname(checkpoint_dir)))
        # ★ 2026-10-03: 同步落 eval_auto_<step>.json 到 ckpt 目录 —— train.py 里
        #   EarlyStopper 的**主数据源**就是这个文件名 (它以前只认 json, 而现代路径
        #   只写 CSV -> 早停静默永不触发, 不报错、只是永远不停)。取 n 最大的 set
        #   (= 最可信) 拍平成扁平指标 (含 skel_iou / lpips, 供 iou_lpips 判据)。
        try:
            import json as _json
            _best = None
            for _s, _v in (res or {}).items():
                if not isinstance(_v, dict):
                    continue
                if _best is None or int(_v.get("n") or 0) > int(
                        res[_best].get("n") or 0):
                    _best = _s
            if _best is not None:
                _flat = {k: v for k, v in res[_best].items()
                         if isinstance(v, (int, float))}
                _flat.update({"step": int(train_steps), "set": _best})
                with open(os.path.join(checkpoint_dir,
                                       f"eval_auto_{int(train_steps)}.json"),
                          "w", encoding="utf-8") as _f:
                    _json.dump(_flat, _f, ensure_ascii=False, indent=2)
                logger.info(
                    f"[in-mem-eval] 早停数据源已落盘 eval_auto_{int(train_steps)}.json "
                    f"(set={_best} n={_flat.get('n')} skel_iou={_flat.get('skel_iou')} "
                    f"lpips={_flat.get('lpips')} ssim={_flat.get('ssim')})")
        except Exception as _je:                              # noqa: BLE001
            logger.info(f"[in-mem-eval] ⚠ eval_auto json 落盘失败(不影响训练/评测): {_je!r}")
        logger.info(
            f"[in-mem-eval] step {train_steps} done in {time.time() - t0:.0f}s: "
            + " | ".join(_fmt_eval(k, v) for k, v in res.items()))
        return res
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"[in-mem-eval] step {train_steps} FAILED: {e}", exc_info=True)
        return None
