# -*- coding: utf-8 -*-
"""dataset.py — 评测**样本映射与缓存**（2026-10-13 [审查 §4.1] 从 src/eval/inference.py 迁出）

审查迁移表: "inference.py:499-701 -> dataset.py：**合并评测样本映射和缓存**" ✓

本模块只负责"评测集 CSV + 图像根 -> 一批固定的 (GT / 条件 / 骨架 / 噪声)":
  · 书家词表校验**放在循环之前**（否则 conds 会比 noise 短, 采样时报一个看不出原因的 size 错 ✗）
  · g 条件三选一: std_soft_root 的 raw mask / skel_latent_shards_dir 的 latent / skel_root PNG
  · 噪声用固定生成器 ⇒ 同一 ckpt 多次评测**完全一致** ✓（评测可复现的前提 ✓）

⚠ 迁移策略（审查"第一阶段只移动逻辑" ✓）: 仍由 src/eval/inference.py **原样再导出** ✓。
依赖: src.utils.latent_dataset.extract_img_id（+ 函数内 import StdSoftGlyphStore ✓）⇒ 无环 ✓
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.utils.latent_dataset import extract_img_id

# img_id 提取失败只警告一次（避免刷屏 ✓）—— 随本函数一起搬来, 仍是唯一一份 ✓
_ID_WARNED = [False]

__all__ = ["make_eval_cache"]


def fixed_eval_noise(rows, latent_channels, latent_spatial):
    """Optional per-image seeds keep nested cohorts paired; old CSVs are unchanged."""
    seeds = [str(row.get('eval_noise_seed', '')).strip() for row in rows]
    if not any(seeds):
        return torch.randn(len(rows), latent_channels, latent_spatial, latent_spatial,
                           generator=torch.Generator().manual_seed(0))
    if not all(seeds):
        raise ValueError('eval_noise_seed must be supplied for every row or none')
    values = [int(seed) for seed in seeds]
    if any(seed < 0 or seed >= 2**63 for seed in values):
        raise ValueError('eval_noise_seed outside [0, 2**63)')
    return torch.stack([torch.randn(latent_channels, latent_spatial, latent_spatial,
                       generator=torch.Generator().manual_seed(seed)) for seed in values])


def make_eval_cache(eval_csv, img_root, skel_root, image_size, n,
                    vae_downscale, latent_channels, scaling_factor,
                    skel_latent_shards_dir=None, callig_id_map=None,
                    callig_script_map=None, std_soft_root=None,
                    std_soft_sigma=0.65):
    """Pre-load N eval samples: GT images + conditions + skels + fixed noise (CPU).

    g 条件三选一: std_soft_root 的 raw mask (N,1,64,64)、
    skel_latent_shards_dir 的 VAE latent (N,4,32,32)、或 skel_root PNG。
    """
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))
    if n > len(rows):
        n = len(rows)
    rows = rows[:n]
    if std_soft_root and skel_latent_shards_dir:
        raise ValueError("std_soft_root and skel_latent_shards_dir are mutually exclusive")
    std_soft_store = None
    if std_soft_root:
        from src.utils.std_soft_glyph import StdSoftGlyphStore
        std_soft_store = StdSoftGlyphStore(
            std_soft_root, is_train=False, eval_sigma=std_soft_sigma,
            preload=True, strict=True)
        std_soft_store.assert_coverage(
            ((int(row["script_id"]), row.get("character", "")) for row in rows),
            label=os.path.basename(eval_csv))
    # ★ 2026-09-17: 书家词表校验**必须放在循环之前**。
    #   原来是在循环里 `continue` 掉不在词表里的行 —— 但 `gts`/`skels`/
    #   `skels_latent`/`noise` 都按 `n` 定长，而 `conds` 是循环里 append 的，
    #   跳过一行就少一个 -> `conds` 比 `noise` 短 -> 采样时 batch 不匹配
    #   （实测 x=16 而 y_callig=15，最后在 factorized_cat 的 cat 里报一个
    #    完全看不出原因的 size 错）。
    #   先全量校验：既一次报出所有越界 id，又保证 conds 与数组严格对齐。
    if callig_id_map is not None:
        _miss = sorted({int(r["calligrapher_id"]) for r in rows
                        if int(r["calligrapher_id"]) not in callig_id_map})
        if _miss:
            raise RuntimeError(
                f"[eval-cache] ✗ {eval_csv} 里有 {len(_miss)} 个书家不在 "
                f"callig_id_map 中: {_miss}\n"
                f"  这些行无法映射到模型的书家表（表大小 {len(callig_id_map)}），"
                f"硬跑会退回原始 id -> 索引越界或用错书家（都是静默的）。\n"
                f"  修法: 换一份只含词表内书家的 eval csv，或把词表补全。\n"
                f"  参考 tools/split_eval_from_50k.py（从训练集切 eval，书家必然在表内）。")
    # (书家×书体) 联合风格词表: 设了则 y_callig 用 pair_id。循环前校验每行 pair 可映射
    # (未见 pair 会回退到该书家默认 pair, 不报错但会告警 —— 避免静默用错风格)。
    if callig_script_map is not None:
        _pm = callig_script_map["pair_map"]
        _miss_pair = sorted({f"{int(r['calligrapher_id'])}:{int(r['script_id'])}"
                             for r in rows
                             if f"{int(r['calligrapher_id'])}:{int(r['script_id'])}" not in _pm})
        if _miss_pair:
            print(f"[eval-cache] ⚠ {len(_miss_pair)} 个 (书家,书体) 对不在词表, "
                  f"将回退到该书家默认 pair: {_miss_pair[:10]}")
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    latent_spatial = image_size // vae_downscale
    gts = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
    conds = []
    # S2: 与 conds 平行的三个 id 列表（书家主效应 / pair 残差 / 书体）
    raw_conds, pair_conds, script_conds = [], [], []
    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)
    if std_soft_store is not None:
        skels_latent = torch.zeros(
            n, 1, std_soft_store.size, std_soft_store.size, dtype=torch.float32)
    elif skel_latent_shards_dir:
        skels_latent = torch.zeros(
            n, latent_channels, latent_spatial, latent_spatial, dtype=torch.float32)
    else:
        skels_latent = None

    # 预建 skel latent shard 索引 (id -> (shard_path, offset))
    skel_id_to_shard = {}
    skel_names = {}
    if skel_latent_shards_dir:
        import glob as _glob
        for sp in sorted(_glob.glob(os.path.join(skel_latent_shards_dir, "shard_*.npz"))):
            with np.load(sp) as d:
                _has_names = "names" in d
                for j, iid in enumerate(d["img_ids"]):
                    _iid = int(iid)
                    if _iid in skel_id_to_shard:
                        _prev_sp, _prev_j = skel_id_to_shard[_iid]
                        raise ValueError(
                            f"[eval-cache] ✗ 致命错误：检测到 ID 碰撞冲突！img_id={_iid}\n"
                            f"  初次出现于: {_prev_sp} (offset {_prev_j})\n"
                            f"  再次出现于: {sp} (offset {j})\n"
                            f"  在分片聚合目录中混入重复号段会导致骨架条件静默错位篡改！"
                        )
                    skel_id_to_shard[_iid] = (sp, j)
                    if _has_names:
                        skel_names[_iid] = str(d["names"][j])

    missing_skel = 0
    for i, row in enumerate(rows):
        p = row["image_path"]
        if img_root and not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        gts[i] = transform(Image.open(p).convert("RGB"))
        # 书家词表收紧: raw id -> 连续索引 (与训练数据层同一张映射表, 见 callig_map.py)
        _cid = int(row["calligrapher_id"])
        if callig_script_map is not None:
            # (书家×书体) 联合风格词表: y_callig = pair_id (与训练数据层同一张表)
            from src.utils.callig_script_map import map_callig_script
            _cid = map_callig_script(int(row["calligrapher_id"]),
                                     int(row["script_id"]), callig_script_map)
        elif callig_id_map is not None:
            # ⚠ 原来是 `callig_id_map.get(_cid, _cid)` —— **不在词表里就退回原始 id**。
            #   两个后果，都是静默的：
            #     ① 原始 id >= 表大小 -> CUDA 索引越界（v13 在 step5000 就这么崩的：
            #        eval 集里有 5 个书家（39/346/401/483/806）不在 50k 的 45 人词表里，
            #        退回原始 id 806 -> y_callig_embedder(806) 而表只有 46 行）
            #     ② 原始 id < 表大小 -> **静默用错书家**，不报错、指标照出
            #   越界行已在**循环之前**全量校验并报错（见本函数开头），这里直接查表。
            _cid = callig_id_map[_cid]
        conds.append((_cid, int(row.get("glyph_id", row.get("character_id", 0)))))
        # ── S2: 主效应 / pair / 书体 三个 id（与训练数据层**同一套映射**）────
        # ⚠ y_callig 在设了 callig_script_map 时装的是 **pair_id(87)**，
        #   hier 模式下主效应表只有 45 行，必须用 callig_raw，否则越界/串书家。
        _raw_cid = int(row["calligrapher_id"])
        if callig_script_map is not None:
            _cm = callig_script_map.get("callig_map") or {}
            _raw_idx = int(_cm.get(str(_raw_cid), _cm.get(_raw_cid, _raw_cid)))
            _pair_idx = int(_cid)
        elif callig_id_map is not None:
            _raw_idx = int(callig_id_map[_raw_cid])
            _pair_idx = _raw_idx
        else:
            _raw_idx = _raw_cid
            _pair_idx = _raw_cid
        raw_conds.append(_raw_idx)
        pair_conds.append(_pair_idx)
        script_conds.append(int(row.get("script_id", 0)))
        # ★ 2026-09-17: img_id 走统一提取（显式列优先 + 正则锚定结尾 + 失败报错）。
        #   原来 `re.search(r"(\d+)\.png", p)` 未锚定、且失败静默给 None ->
        #   非数字文件名会静默不查 skel（g=ZERO）而不是报错（见 docs/system/70 §1.2）。
        try:
            img_id = extract_img_id(row, where="eval_cache")
        except ValueError as _e:
            if not _ID_WARNED[0]:
                print(f"[eval-cache] ⚠ {_e}")
                _ID_WARNED[0] = True
            img_id = None
        if std_soft_store is not None:
            skels_latent[i] = std_soft_store.get(
                int(row["script_id"]), row.get("character", ""))
        elif img_id is not None and skel_names:
            # 与 latent_dataset 同一套内容绑定校验: shard 里的 names 必须等于本行的骨架图名。
            # 没有这道闸，小数据集(如 few-shot)指向大库时会号段撞车 —— 查得到骨架但是
            # 别的字的，评测照样出分，只是分全是错的（2026-09-21 实测作废一整轮实验）。
            _want = os.path.basename(row.get("std_path") or p)
            _got = skel_names.get(int(img_id))
            if _got and _want and _got != _want:
                raise ValueError(
                    f"[eval-cache] skel latent 与 CSV 对不上: img_id={img_id} "
                    f"需要 {_want} 但 shard 里是 {_got} —— "
                    f"检查 eval_skel_latent_shards_dir 是否指向本评测集自己的 shard")
        if std_soft_store is not None:
            pass
        elif img_id is not None and skels_latent is not None:
            if img_id in skel_id_to_shard:
                sp, j = skel_id_to_shard[img_id]
                with np.load(sp, mmap_mode="r") as d:      # ★ mmap: 别解压整个 shard
                    skels_latent[i] = torch.from_numpy(np.array(d["latents"][j], copy=True)).float()
            else:
                missing_skel += 1
        elif img_id is not None and skel_root:
            sk = Image.open(os.path.join(skel_root, f"{img_id}.png")).convert("L")
            sk = sk.resize((image_size, image_size), Image.NEAREST)
            skels[i, 0] = torch.from_numpy(np.asarray(sk, np.float32) / 255.0)
    noise = fixed_eval_noise(rows, latent_channels, latent_spatial)
    if skels_latent is not None and missing_skel:
        # ★ 2026-09-17: 从"只打一行 WARNING 继续跑"改为**启动即失败**。
        #   原行为的问题（doc68 §2.2 实际踩过）: shard 目录配错 -> strict 命中 0/237
        #   -> g 全零 -> 字条件完全失效，但评测**照跑不误**，指标看着正常。
        #   早失败 10 秒，胜过跑完 50 步再发现。
        _rate = missing_skel / max(n, 1)
        msg = (f"[eval-cache] ✗ {missing_skel}/{n} ({_rate*100:.1f}%) 样本在 "
               f"{skel_latent_shards_dir!r} 里找不到 skel latent -> 这些样本 g=ZERO"
               f"（字条件失效）")
        if _rate > 0.02:
            raise RuntimeError(
                msg + "\n  覆盖率 <98% 说明 shard 目录配错了（常见：把训练 shard 当成 "
                      "eval shard，或 eval csv 的 id 与 shard 不是同一套）。\n"
                      "  修法: 为 eval csv 单独建 shards（见 config 的 "
                      "eval_skel_latent_shards_dir），或确认 id 一致。\n"
                      "  若确实要容忍少量缺失，请显式降低阈值——不要静默继续。")
        print(msg + "  (比例低, 继续)", flush=True)
    return {"rows": rows, "gts": gts, "conds": conds, "noise": noise, "skels": skels,
            "hier_conds": (raw_conds, pair_conds, script_conds),
            "skels_latent": skels_latent, "missing_skel": missing_skel,
            "n": n, "latent_channels": latent_channels,
            "latent_spatial": latent_spatial, "scaling_factor": scaling_factor,
            "img_root": img_root, "skel_root": skel_root, "image_size": image_size,
            "g_kind": "std_soft" if std_soft_store is not None else
                      ("latent" if skel_latent_shards_dir else "png")}
