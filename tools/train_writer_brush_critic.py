#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_writer_brush_critic.py —— 双头条件判别器（**latent ✓** 训练中不挂 decoder ✓）。

★ 大部分采用**刚才验证有效**的方法（用户指示 ✓）:
    ① **26 类 CE**（23 书家槽 + 3 std 虚拟书家槽 ✓）—— 像素版已验证:
        作品级留出 G1 2AFC **0.748 > 0.65** ✓ std acc 0.92 ✓
        （InfoNCE 已被否定 ✗ —— 分类任务就该 CE ✓ 用户指出 ✓）
    ② **同字难负构造**（扫描验证的关键 ✓）: pair 正负**同字·同书体·异书家** ✓
        ⇒ 方案的 L_pair ✓（softplus margin, m=0.3 ✓）
    ③ 作品级留出 ✓（记忆不算数 ✓）④ model.eval() ✓（BN 坑 ✓）
    ⑤ **std 当独立类**（用户原始想法 ✓ 已验证 ✓）⇒ 书家 vs std 分离进 CE ✓

＋ 方案新增: **Brush 头**（真迹 vs std 二分类 ✓）⇒ 第二阶段 AwaySTD 的监督来源 ✓
    L = L_slot(26类CE) + 0.2·L_pair + 0.2·L_brush ✓（方案初始权重 ✓）

验收四项（方案 §五 ✓）: T1 同字异书家 2AFC≥0.65（留出 ✓）| T2 跨字符检索 mAP ✓
    | T3 brush 正负率+白/噪探针 ✓ | T4 作品级留出（inherent ✓）
基线口径（用户指出 ✓）: **同时报 1/K（槽 K 路）与 0.5（2AFC）** ✓
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEP = "·"


class WriterBrushCritic(nn.Module):
    """z[B,4,32,32] + g[B,1,64,64] + script ⇒ slot_logits[K] + brush 标量 + feat ✓。"""

    def __init__(self, n_slots, n_scripts, d=256):
        super().__init__()
        self.z_stem = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),    # 16
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU())   # 8
        self.g_stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(),    # 32
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),   # 16
            nn.Conv2d(32, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU())   # 8
        self.trunk = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.Conv2d(192, 192, 3, 2, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.Conv2d(192, 192, 3, 1, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.script_emb = nn.Embedding(n_scripts, 32)
        self.fuse = nn.Sequential(nn.Linear(224, 384), nn.GELU(),
                                  nn.Linear(384, 256), nn.GELU())
        self.slot_head = nn.Linear(256, n_slots)
        self.brush_head = nn.Linear(256, 1)

    def forward(self, z, g, script):
        zf, gf = self.z_stem(z), self.g_stem(g)
        h = self.trunk(torch.cat([zf, gf], 1)).flatten(1)
        h = self.fuse(torch.cat([h, self.script_emb(script)], 1))
        return {"slot_logits": self.slot_head(h),
                "brush_logit": self.brush_head(h).squeeze(-1),
                "feat": h}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scratch/disc_lat_full.npz")
    ap.add_argument("--out", default="assets/writer_brush_critic.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--w-pair", type=float, default=0.2)
    ap.add_argument("--w-brush", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--std-frac", type=float, default=0.25)
    ap.add_argument("--pair-frac", type=float, default=0.5)
    ap.add_argument("--holdout-work-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    d = np.load(args.data, allow_pickle=True)
    z_all = torch.from_numpy(d["z"].astype(np.float32))
    writer_all = [str(x) for x in d["writer"]]
    script_all = [str(x) for x in d["script"]]
    char_all = [str(x) for x in d["char"]]
    work_all = [str(x) for x in d["work"]]
    std_idx = d["std_idx"].astype(np.int64)
    z_std = torch.from_numpy(d["z_std"].astype(np.float32))
    g_std = torch.from_numpy(d["g_std"].astype(np.float32)) / 255.0
    std_script = [str(x) for x in d["std_script"]]
    probe_names = [str(x) for x in d.get("probe_names", [])]
    z_probe = (torch.from_numpy(d["z_probe"].astype(np.float32))
               if "z_probe" in d else torch.zeros(0, 4, 32, 32))
    scripts = [str(x) for x in d["scripts"]]
    s2i = {s: i for i, s in enumerate(scripts)}
    N, M = len(writer_all), len(std_script)

    # ★ 槽位表 = 验证有效的 26 类设计 ✓
    writer_slots = sorted({writer_all[i] + SEP + script_all[i] for i in range(N)})
    std_slots = sorted({"std" + SEP + s for s in scripts})
    slots = writer_slots + std_slots
    slot2i = {k: i for i, k in enumerate(slots)}
    K = len(slots)
    print("[slots] %d = %d 书家槽 + %d std 槽 ✓" % (K, len(writer_slots), len(std_slots)),
          flush=True)

    works = sorted(set(work_all))
    rng.shuffle(works)
    n_hold = max(1, int(len(works) * args.holdout_work_frac))
    hold = set(works[:n_hold])
    tr_idx = [i for i in range(N) if work_all[i] not in hold]
    ev_idx = [i for i in range(N) if work_all[i] in hold]
    print("[split] 作品 %d ⇒ 训练 %d / 留出 %d（样本 %d/%d ✓）"
          % (len(works), len(works) - n_hold, n_hold, len(tr_idx), len(ev_idx)), flush=True)

    slot_id = np.array([slot2i[writer_all[i] + SEP + script_all[i]] for i in range(N)],
                       np.int64)
    sid_all = np.array([s2i[script_all[i]] for i in range(N)], np.int64)
    sid_std = np.array([s2i[s] for s in std_script], np.int64)

    # ── 同字·同书体·异书家 pair（训练 split 内 ✓ = 验证有效的难负构造 ✓）──
    def build_pairs(idxs):
        by_key = defaultdict(list)
        for i in idxs:
            by_key[(script_all[i], char_all[i])].append(i)
        out = []
        for idxs2 in by_key.values():
            byw = defaultdict(list)
            for i in idxs2:
                byw[slot_id[i]].append(i)
            ws = sorted(byw)
            for a in range(len(ws)):
                for b in range(a + 1, len(ws)):
                    out.append((rng.choice(byw[ws[a]]), rng.choice(byw[ws[b]])))
        rng.shuffle(out)
        return out

    pairs = build_pairs(tr_idx)
    pairs_e = build_pairs(ev_idx)
    print("[pair] 训练 pair %d（覆盖 %d 字 ✓）| 留出 pair %d ✓"
          % (len(pairs), len({(script_all[i], char_all[i]) for i, _ in pairs}),
             len(pairs_e)), flush=True)

    model = WriterBrushCritic(K, len(scripts)).to(dev)
    print("[model] %s 参数" % format(sum(p.numel() for p in model.parameters()), ","),
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def g_of(rows_idx):
        return g_std[torch.from_numpy(np.asarray(rows_idx))].to(dev).unsqueeze(1)

    def run_step():
        n_std = int(args.batch * args.std_frac)
        n_pair = (args.batch - n_std) // 4
        idx_r = []
        npair_used = 0
        if pairs:
            for _ in range(n_pair):
                a, b = pairs[rng.randrange(len(pairs))]
                idx_r += [a, b]
                npair_used += 1
        while len(idx_r) < args.batch - n_std:
            idx_r.append(tr_idx[rng.randrange(len(tr_idx))])
        idx_r = idx_r[:args.batch - n_std]
        idx_s = [rng.randrange(M) for _ in range(n_std)]
        n_ood = min(len(z_probe), max(1, n_std // 4)) if len(z_probe) else 0
        nr = len(idx_r)

        # ★★ brush 的 OOD 负样本（白/噪探针 latent ✓）混入 std 负样本区 ✓
        #    （上一版编辑的锚点没匹配 ⇒ 静默失效 ✗✗ ⇒ 探针从未进训练 ⇒ T3 全灭 ✗
        #      教训: 长 anchor 的 s.replace 必须 assert 才算数 ✓）
        z_parts = [z_all[idx_r].to(dev), z_std[idx_s].to(dev)]
        y_parts = [slot_id[idx_r],
                   np.array([slot2i["std" + SEP + std_script[j]] for j in idx_s])]
        si_parts = [sid_all[idx_r], sid_std[idx_s]]
        g_rows_parts = [np.clip(std_idx[idx_r], 0, None), np.asarray(idx_s)]
        for oi in range(n_ood):                     # OOD 探针 ⇒ std 槽 + brush 负 ✓
            z_parts.append(z_probe[oi:oi + 1].to(dev))
            y_parts.append(np.array([slot2i["std" + SEP + scripts[oi % len(scripts)]]]))
            si_parts.append(np.array([oi % len(scripts)]))
            g_rows_parts.append(np.array([rng.randrange(len(std_script))]))
        zi = torch.cat(z_parts, 0)
        y = torch.from_numpy(np.concatenate(y_parts)).to(dev)
        si = torch.from_numpy(np.concatenate(si_parts)).to(dev)
        g = g_of(np.concatenate(g_rows_parts))
        out = model(zi, g, si)

        l_slot = F.cross_entropy(out["slot_logits"], y)
        l_pair = torch.zeros((), device=dev)
        if npair_used:
            logp = F.log_softmax(out["slot_logits"][:nr], dim=-1)
            pa = torch.arange(npair_used, device=dev) * 2
            pb = pa + 1
            wa = y[pa]
            l_pair = F.softplus(args.margin + logp[pb, wa] - logp[pa, wa]).mean()
        l_brush = 0.5 * (F.softplus(-out["brush_logit"][:nr]).mean()
                         + F.softplus(out["brush_logit"][nr:]).mean())
        loss = l_slot + args.w_pair * l_pair + args.w_brush * l_brush
        acc = (out["slot_logits"].argmax(1) == y).float().mean()
        return loss, {"slot": float(l_slot), "pair": float(l_pair),
                      "brush": float(l_brush), "acc": float(acc)}

    steps_per_epoch = max(1, len(tr_idx) // args.batch)
    total = args.epochs * steps_per_epoch
    print("[train] %d ep × %d st = %d ✓" % (args.epochs, steps_per_epoch, total), flush=True)
    t0 = time.time()
    for step in range(1, total + 1):
        loss, p = run_step()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, total // 10) == 0 or step == 1:
            print("  [step %5d/%d] L %.4f (slot %.4f pair %.4f brush %.4f) acc %.3f | %.0fs"
                  % (step, total, float(loss), p["slot"], p["pair"], p["brush"],
                     p["acc"], time.time() - t0), flush=True)

    # ── 验收（作品级留出 ✓ + model.eval ✓）──
    model.eval()
    print("[eval] model.eval() ✓", flush=True)
    with torch.no_grad():
        def embed(idxs, bs=512):
            fs, lp = [], []
            for s in range(0, len(idxs), bs):
                ii = idxs[s:s + bs]
                o = model(z_all[ii].to(dev), g_of(np.clip(std_idx[ii], 0, None)),
                          torch.from_numpy(sid_all[ii]).to(dev))
                fs.append(F.normalize(o["feat"], dim=-1).cpu().numpy())
                lp.append(F.log_softmax(o["slot_logits"], -1).cpu().numpy())
            return np.concatenate(fs), np.concatenate(lp)

        F_e, LP_e = embed(ev_idx)
        y_e = slot_id[ev_idx]
        pos_of = {gi: k for k, gi in enumerate(ev_idx)}
        sc_pos, sc_neg = [], []
        for a, b in pairs_e:
            if a in pos_of and b in pos_of:
                sc_pos.append(LP_e[pos_of[a], slot_id[a]])
                sc_neg.append(LP_e[pos_of[b], slot_id[a]])   # 负样本按**正样本槽**评判 ✓
        acc = float(np.mean([p > n for p, n in zip(sc_pos, sc_neg)])) if sc_pos \
            else float("nan")
        ci = [float("nan")] * 2
        if sc_pos:
            boots = [float(np.mean([sc_pos[i] > sc_neg[i] for i in
                                    np.random.randint(0, len(sc_pos), len(sc_pos))]))
                     for _ in range(1000)]
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        pred = LP_e.argmax(1)
        top1 = float((pred == y_e).mean())
        top5 = float(np.mean([yy in pp for yy, pp in zip(y_e, np.argsort(-LP_e, 1)[:, :5])]))
        S = F_e @ F_e.T
        np.fill_diagonal(S, -2)
        order = np.argsort(-S, 1)
        sc_e = np.array([script_all[i] for i in ev_idx])
        y_arr = np.asarray(y_e)
        maps, maps_base = [], []
        for qi in range(len(ev_idx)):
            rel = (y_arr[order[qi]] == y_arr[qi]).astype(float)
            if rel.sum() >= 1:
                prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
                maps.append(float((prec * rel).sum() / rel.sum()))
            rel_b = (sc_e[order[qi]] == sc_e[qi]).astype(float)
            if rel_b.sum() >= 1:
                prec = np.cumsum(rel_b) / (np.arange(len(rel_b)) + 1)
                maps_base.append(float((prec * rel_b).sum() / rel_b.sum()))
        b_real = []
        for s in range(0, len(ev_idx), 512):
            ii = ev_idx[s:s + 512]
            o = model(z_all[ii].to(dev), g_of(np.clip(std_idx[ii], 0, None)),
                      torch.from_numpy(sid_all[ii]).to(dev))
            b_real.append(o["brush_logit"].cpu().numpy())
        b_real = np.concatenate(b_real)
        o_std = model(z_std.to(dev), g_std.to(dev).unsqueeze(1),
                      torch.from_numpy(sid_std).to(dev))
        b_std = o_std["brush_logit"].cpu().numpy()
        probes = {}
        try:
            from PIL import Image
            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained("data/pretrained/sd-vae-ft-ema").to(dev).eval()

            def zof(img):
                x = (torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1)
                     .float().unsqueeze(0).to(dev) / 127.5 - 1.0)
                with torch.no_grad():
                    return vae.encode(x).latent_dist.mode() * 0.18215
            white = Image.new("RGB", (256, 256), (255, 255, 255))
            noise = Image.fromarray(np.random.RandomState(0).randint(
                0, 256, (256, 256, 3)).astype(np.uint8))
            g0 = g_std[:1].to(dev).unsqueeze(1)
            s0 = torch.zeros(1, dtype=torch.long, device=dev)
            probes["white_b"] = float(model(zof(white), g0, s0)["brush_logit"][0])
            probes["noise_b"] = float(model(zof(noise), g0, s0)["brush_logit"][0])
            del vae
            torch.cuda.empty_cache()
        except Exception as e:                                          # noqa: BLE001
            probes["err"] = str(e)[:100]
    res = {"T1_pair_2afc": acc, "T1_ci95": ci, "n_pairs_holdout": len(sc_pos),
           "T1_slot_top1": top1, "T1_slot_top5": top5,
           "chance_1ofK": 1.0 / K, "chance_2afc": 0.5,
           "T2_map": float(np.mean(maps)) if maps else None,
           "T2_map_same_script_base": float(np.mean(maps_base)) if maps_base else None,
           "T3_brush_real_pos": float((b_real > 0).mean()),
           "T3_brush_std_neg": float((b_std < 0).mean()), "T3_probes": probes,
           "n_holdout": len(ev_idx)}
    print("\n===== 验收（作品级留出 ✓）=====")
    print("  T1 同字异书家 2AFC = %.4f CI95=%s（随机 0.5 / 门槛 0.65 ✓）"
          % (acc, ["%.3f" % c for c in ci]))
    print("  T1 槽 K 路 top1 = %.4f（随机 1/%d=%.4f ✓ 用户口径 ✓）top5 %.4f"
          % (top1, K, 1.0 / K, top5))
    print("  T2 检索 mAP = %s（同书体基线 %s）"
          % (("%.4f" % res["T2_map"]) if res["T2_map"] else "NA",
             ("%.4f" % res["T2_map_same_script_base"])
             if res["T2_map_same_script_base"] else "NA"))
    print("  T3 brush: 真迹 b>0 %.3f | std b<0 %.3f | 探针 %s"
          % (res["T3_brush_real_pos"], res["T3_brush_std_neg"], probes))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "slots": slots, "scripts": scripts,
                "results": res}, args.out)
    json.dump(res, open(args.out.replace(".pt", "_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print("[done] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
