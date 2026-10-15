# -*- coding: utf-8 -*-
"""src/loss/critic_loss.py —— v85B/C 生成端辅助 Loss（冻结双头 critic ✓ 方案 §六 ✓）。

L_writer^G = CE(critic.slot_logits(ẑ₀, g, s), 目标书家槽)     ← 生成图要被认对书家 ✓
L_awaySTD^G = softplus(−critic.brush_logit(ẑ₀, g, s))          ← 生成图要有真迹笔墨 ✓
L_relSTD^G  = softplus[ ρ·clip(b_real−b_std,0,M) − (b_gen−b_std) ]   （可选 ✓）

约束 ✓: 全程 latent（ẑ₀ 直接喂 critic ✓ 训练中**不挂 decoder** ✓）
        critic 冻结 eval() ✓ **不套 no_grad** ✓（否则不回传 ✗ 方案警告 ✓）
        critic(ẑ₀) 的梯度只进 DiT ✓（critic 参数 requires_grad_(False) ✓）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticLoss(nn.Module):
    """加载冻结双头 critic，提供 per-sample 的 writer CE 与 brush 项 ✓。

    坐标契约（必须与训练数据一致 ✓）:
        z ∈ [B,4,32,32]  **已乘 sf=0.18215** ✓（训练 latent 的同一坐标 ✓）
        g ∈ [B,1,64,64] ∈ [0,1]（std 条件 ✓ 固定 σ blur ✓ —— 即 npz 的 g_std ✓）
        script ∈ [B] int64
        y_slot ∈ [B] int64（**书家槽 id**，不含 std 槽 ✓ —— 生成图不该是 std ✓）
    """

    def __init__(self, ckpt: str, device: str = "cuda"):
        super().__init__()
        import importlib
        _twb = importlib.import_module('tools.train_writer_brush_critic')
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.slots = [str(x) for x in ck["slots"]]
        self.scripts = [str(x) for x in ck["scripts"]]
        s2i = {s: i for i, s in enumerate(self.scripts)}
        self.model = _twb.WriterBrushCritic(len(self.slots), len(self.scripts))
        self.model.load_state_dict(ck["model"])
        self.model.requires_grad_(False)          # 冻结 ✓
        self.model.eval()                         # BN running stats ✓
        self.s2i = s2i
        # std 条件查找表: (script,char) → g_std 行 ✓ 由外部 attach ✓
        self._g_lookup = None                     # (g_std[T,64,64], key→row dict)
        self.to(device)

    def build_id2slot(self, csv_path, callig_remap=None, font_remap=None):
        """(callig_raw_id, script_remap_id) → slot 字符串 ✓。

        数据集的 y_callig_raw/y_script 是**重映射后的连续 id** ✓（不是 CSV 原始 id ✗）
        必须传入与数据集相同的callig/font remap。无remap时书家按原始ID排序，
        书体保留原始ID；绝不按CSV出现顺序推断连续ID。
        """
        import csv
        import json
        def load_map(value):
            if isinstance(value, str) and value:
                with open(value, encoding='utf-8') as f:
                    value = json.load(f)
            return {int(k): int(v) for k, v in value.items()} if value else None
        cm, sm = load_map(callig_remap), load_map(font_remap)
        with open(csv_path, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        # Dataset defaults use sorted raw writer IDs, NOT CSV appearance order.
        if cm is None:
            cm = {raw: i for i, raw in enumerate(sorted({int(r['calligrapher_id']) for r in rows}))}
        c2i, s2i = {}, {}
        for r in rows:
            w, s = r['calligrapher'].strip(), r['script'].strip()
            wi = cm[int(r['calligrapher_id'])]
            si = sm[int(r['script_id'])] if sm is not None else int(r['script_id'])
            if (w in c2i and c2i[w] != wi) or (s in s2i and s2i[s] != si):
                raise ValueError('inconsistent dataset writer/script mapping')
            c2i[w], s2i[s] = wi, si
        if len(set(c2i.values())) != len(c2i) or len(set(s2i.values())) != len(s2i):
            raise ValueError('non-injective dataset writer/script mapping')
        lut = {}
        for k in self.slots:
            if k.startswith('std·'):
                continue
            w, s = k.split('·')
            if w in c2i and s in s2i:
                lut[(c2i[w], s2i[s])] = k
        self.id2slot = lut
        self.id2callig = c2i
        self.id2script = s2i
        return lut

    def attach_std_bank(self, npz: str):
        d = np.load(npz, allow_pickle=True)
        self.g_std_bank = torch.from_numpy(d["g_std"].astype(np.float32)) / 255.0
        keys = [str(s) + "·" + str(c) for s, c in zip(d["std_script"], d["std_char"])]
        self._key2row = {k: i for i, k in enumerate(keys)}

    def _g_for(self, script_names, chars, device):
        """strict 查表 ✓（2026-10-14 [重构 P0-3] ✗: 旧实现 `get(...,-1)` + `np.clip`
        把**查不到的 (书体,字符) 悄悄映射到 bank 第 0 行** ✗✗ —— 生成「家」却喂了
        别的字的 STD ✗ ⇒ 判别器收到的 g 与生成内容不符 ✗。

        新语义: 任何缺失 ⇒ **ValueError 并列出缺失样本** ✗（宁可不跑, 不喂错字 ✗）。
        调用方应在训练前用 `coverage_audit()` 做一次性覆盖率审计 ✓。
        """
        rows = np.array([self._key2row.get(s + "·" + c, -1)
                         for s, c in zip(script_names, chars)], np.int64)
        bad = np.nonzero(rows < 0)[0]
        if bad.size:
            miss = sorted({(script_names[i], chars[i]) for i in bad.tolist()})
            raise KeyError(
                f"[critic] std bank 缺 {bad.size}/{len(rows)} 个 (script,char) ✗ "
                f"示例={miss[:6]} —— strict_lookup 拒绝用第 0 行冒充 ✗")
        return self.g_std_bank[torch.from_numpy(rows)].to(device).unsqueeze(1)

    def _g_for_rows(self, std_rows, device):
        """直接给 std bank 行号 ✓ —— **同样 strict** ✗（旧 np.clip 同罪 ✗）。"""
        rows = np.asarray(std_rows, np.int64)
        bad = np.nonzero((rows < 0) | (rows >= len(self.g_std_bank)))[0]
        if bad.size:
            raise IndexError(
                f"[critic] std_rows 越界 {bad.size}/{len(rows)} 个 "
                f"(示例 rows={rows[bad[:6]].tolist()} bank={len(self.g_std_bank)}) ✗ —— "
                f"strict_lookup 拒绝 clip 到第 0 行冒充 ✗")
        return self.g_std_bank[torch.from_numpy(rows)].to(device).unsqueeze(1)

    @torch.no_grad()
    def coverage_audit(self, script_names, chars) -> Dict[str, int]:
        """一次性覆盖率审计 ✓（训练/评测前调用 ✗ 缺失即报 ✗ 不静默 ✗）。

        返回 {"total", "missing"}; missing>0 时**直接抛 KeyError** ✗。
        """
        total = len(script_names)
        rows = np.array([self._key2row.get(s + "·" + c, -1)
                         for s, c in zip(script_names, chars)], np.int64)
        miss = int((rows < 0).sum())
        if miss:
            bad = np.nonzero(rows < 0)[0]
            examples = sorted({(script_names[i], chars[i]) for i in bad.tolist()})
            raise KeyError(
                f"[critic] 覆盖率审计失败: {miss}/{total} 缺失 ✗ 示例={examples[:8]}")
        return {"total": total, "missing": 0}

    def set_char_lookup(self, csv_path, roots, scripts, chars_vocab=None):
        """建 (script_id, char_id) → std_bank 行 的映射 ✓。

        train.py 的 batch 给的是**整数索引** ✓（y_script/y_char ✗ 不是字符串 ✓）
        ⇒ 必须在 train.py 侧用数据集 CSV 建一次 id→str 映射 ✓ 再查 std 行 ✓。
        """
        import csv
        s2i = {s: i for i, s in enumerate(scripts)}
        lut = {}
        for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
            s = (r.get("script") or "").strip()
            c = (r.get("character") or "").strip()
            # ★ 2026-10-14 [重构 P1]: 键改为 (script_id, Unicode codepoint) ✓
            #   旧 hash(c) 依赖 PYTHONHASHSEED ✗ 跨进程不保证一致 ✗ 也不是稳定 ID ✗
            cp = ord(c) if len(c) == 1 else -1     # 多字符/空 → 无法定位 ⇒ -1 缺失
            k = (s2i.get(s, -1), cp)
            if k not in lut:
                lut[k] = self._key2row.get(s + "·" + c, -1)
        self._lut = lut

    def forward_idx(self, z_pred, script_ids, char_ids, std_rows, target_slots, device):
        """**索引版** forward ✓（train.py 传整数张量 ✓ —— 避免字符串换转 ✗）。"""
        g = self._g_for_rows(std_rows, device)
        si = torch.as_tensor(script_ids, dtype=torch.long, device=device)
        ys = torch.as_tensor(target_slots, dtype=torch.long, device=device)
        out = self.model(z_pred, g, si)           # ★ 无 no_grad ✓（冻结但可导 ✓）
        l_writer = F.cross_entropy(out["slot_logits"], ys)
        l_away = F.softplus(-out["brush_logit"]).mean()
        # ★ 2026-10-14 [重构 P1-6]: 返回 **detached GPU tensor** ✗
        #   旧实现 float(detach()) 每步强制 CUDA→CPU 同步 ✗（训练热路径的同步点 ✗）
        #   ⇒ 调用方只在日志周期才 .item() 转 CPU 数值 ✓
        return l_writer, l_away, {"writer": l_writer.detach(),
                                  "away": l_away.detach()}

    @torch.no_grad()
    def b_real_b_std(self, z_real, script_names, chars, device):
        """供 relativeSTD 变体用的 (b_real, b_std) ✓（一次性、无梯度 ✓）。"""
        g = self._g_for(script_names, chars, device)
        si = torch.tensor([self.s2i[s] for s in script_names], device=device)
        out = self.model.module if hasattr(self.model, "module") else self.model
        b_r = out(z_real.to(device), g, si)["brush_logit"]
        return b_r

    def forward_g(self, z_pred, g, script_ids, target_slots):
        """**直接给 g 张量** 的 forward ✓（train.py 用 batch['std_soft'] ✓）。

        g: [B,1,64,64] ∈[0,1] ✓（std_soft ✓ 与 critic 的 g_std 同源 ✓）
        script_ids/target_slots: LongTensor ✓（id 空间 ✓）
        """
        si = torch.as_tensor(script_ids, dtype=torch.long,
                             device=z_pred.device)
        ys = torch.as_tensor(target_slots, dtype=torch.long,
                             device=z_pred.device)
        out = self.model(z_pred, g, si)           # ★ 无 no_grad ✓（冻结但可导 ✓）
        l_writer = F.cross_entropy(out["slot_logits"], ys)
        l_away = F.softplus(-out["brush_logit"]).mean()
        return l_writer, l_away, {"writer": l_writer.detach(),
                                  "away": l_away.detach()}   # P1-6 ✓ detached tensor

    def forward(self, z_pred, script_names, chars, target_slots, device):
        """返回 (loss_writer, loss_away, parts)。**必须有梯度** ✓（z_pred 带 graph ✓）。"""
        g = self._g_for(script_names, chars, device)
        si = torch.tensor([self.s2i[s] for s in script_names], device=device)
        ys = torch.tensor(target_slots, dtype=torch.long, device=device)
        out = self.model(z_pred, g, si)           # ★ 无 no_grad ✓（冻结但可导 ✓）
        l_writer = F.cross_entropy(out["slot_logits"], ys)
        l_away = F.softplus(-out["brush_logit"]).mean()
        return l_writer, l_away, {"writer": l_writer.detach(),
                                  "away": l_away.detach()}   # P1-6 ✓ detached tensor
