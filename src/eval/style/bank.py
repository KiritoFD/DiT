# -*- coding: utf-8 -*-
"""Style Reference Bank —— 真实真迹的风格参考库（按 writer × script × character × work 索引 ✓）。

审查要求 ✓
    · 参考图必须**来自不同作品、不同字符**（否则高检索率可能只是同一个字/同一次扫描带来的 ✓）
    · 每个有效书家—书体槽位尽量 30–100 张（不足则**显式标记为不可计算**，不硬算 ✓）
    · 作品 id 取 `src_image_path` 的**所在目录**（同一碑帖/作品 ⇒ 同组 ✓）

本模块**只处理索引与划分**，不碰像素 ✓（可离线单测 ✓）。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_KEYS = {"楷": "kai", "行": "xing", "隶": "li", "草": "cao", "篆": "zhuan"}


@dataclass(frozen=True)
class RefRow:
    path: str
    src_path: str
    writer: str
    script: str
    char: str
    work: str
    split: str                      # "ref" | "query" | "excluded"
    aug: str = ""

    @property
    def slot(self) -> Tuple[str, str]:
        return (self.writer, self.script)


GENERIC_DIRS = {"imgs", "img", "images", "image", "std", "out", "gen", "samples"}


def work_id(src_image_path: str, image_path: str = "", depth: int = 2) -> str:
    """作品 id = `src_image_path` 的**碑帖层**目录 ✓。

    ⚠ 第一版取"所在目录名" ⇒ 真实 CSV 是 `data/fame-kxl-tj-px60/imgs/287458.png`,
      取到的是 **`imgs`** ✗ ⇒ 所有行同一个 id ⇒ "排除同作品"完全失效 ✗✗（实测暴露 ✓）。
    ⇒ 改为取**上两层**（`fame-kxl-tj-px60/imgs`）✓, 并跳过 `imgs/std` 这类通名 ✓。
    """
    s = (src_image_path or image_path or "").strip().replace("\\", "/")
    if not s:
        return "unknown"
    parts = [p for p in os.path.dirname(s).split("/") if p]
    if not parts:
        return os.path.basename(s) or "unknown"
    # 去掉末尾的通名目录（imgs/…）⇒ 保留真正的作品层 ✓
    while len(parts) > 1 and parts[-1].lower() in GENERIC_DIRS:
        parts = parts[:-1]
    return "/".join(parts[-depth:]) if len(parts) >= 2 else parts[-1]


def resolve(path: str, roots: Sequence[str] = ()) -> Optional[str]:
    """把 CSV 里的相对路径解析到实际存在的文件 ✓（多候选根, 找不到返回 None ✓）。"""
    p = (path or "").strip()
    if not p:
        return None
    cands = [p] + [os.path.join(r, p) for r in roots if r]
    cands += [os.path.join(r, os.path.basename(p)) for r in roots if r]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def build_bank(csv_path: str, *, roots: Sequence[str] = (), drop_aug: bool = True,
               exclude_ids: Sequence[str] = (), max_rows: Optional[int] = None,
               verbose: bool = True) -> List[RefRow]:
    """从训练 CSV 建参考库（保留全部候选, 由调用方决定 ref/query 划分 ✓）。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    ex = set(exclude_ids or ())
    out: List[RefRow] = []
    n_missing = 0
    for r in rows:
        aug = (r.get("aug") or "").strip()
        if drop_aug and aug:
            continue
        src = (r.get("src_image_path") or "").strip()
        if src and (src in ex or os.path.basename(src) in ex):
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            n_missing += 1
            continue
        out.append(RefRow(path=p, src_path=src,
                          writer=(r.get("calligrapher") or "").strip(),
                          script=(r.get("script") or "").strip(),
                          char=(r.get("character") or "").strip(),
                          work=work_id(src, r.get("image_path", "")), split="ref", aug=aug))
        if max_rows and len(out) >= max_rows:
            break
    if verbose:
        slots: Dict[Tuple[str, str], int] = {}
        for x in out:
            slots[x.slot] = slots.get(x.slot, 0) + 1
        print(f"[bank] 可用 {len(out)} 张（跳过缺图 {n_missing}）| 槽位 {len(slots)} 个 ✓")
        for k in sorted(slots):
            print(f"    {k[0]:<8} {k[1]:<3} {slots[k]:>6}")
    return out


def split_by_work(bank: Sequence[RefRow], *, query_frac: float = 0.3, seed: int = 0,
                  min_ref: int = 4) -> Tuple[List[RefRow], List[RefRow]]:
    """按**作品**划分 ref/query ✓ —— 保证 query 的作品在 gallery 中**不出现** ✓。

    同一槽位内: 作品数 ≥3 时才划出 query（否则该槽位整体留作参考 ✓，避免不可算 ✓）。
    """
    import random
    rng = random.Random(seed)
    by_slot: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for i, r in enumerate(bank):
        by_slot.setdefault(r.slot, {}).setdefault(r.work, []).append(i)
    ref, qry = [], []
    for slot, works in by_slot.items():
        ws = sorted(works)
        if len(ws) >= 3:
            rng.shuffle(ws)
            n_q = max(1, int(round(len(ws) * query_frac)))
            qset = set(ws[:min(n_q, len(ws) - 1)])
        else:
            qset = set()
        for w, idxs in works.items():
            for i in idxs:
                (qry if w in qset else ref).append(bank[i])
    return ref, qry


def cap_balanced(rows: Sequence[RefRow], n: int) -> List[RefRow]:
    """把参考库压到 n 张，**按槽位均衡**（轮转取样 ✓）。

    ⚠ 直接截断 CSV 会**整族丢掉后面的书家** ✗（CSV 按书家排序 ✓）⇒ 检索候选书家数会变 ✗。
      轮转取样保证每个槽位按比例保留 ✓，且不改变 writer 集合 ✓。
    """
    if n <= 0 or len(rows) <= n:
        return list(rows)
    by_slot: Dict[Tuple[str, str], List[RefRow]] = {}
    for r in rows:
        by_slot.setdefault(r.slot, []).append(r)
    out: List[RefRow] = []
    i = 0
    while len(out) < n:
        added = False
        for slot in sorted(by_slot):
            lst = by_slot[slot]
            if i < len(lst):
                out.append(lst[i])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        i += 1
    return out


def slot_counts(rows: Sequence[RefRow]) -> Dict[Tuple[str, str], int]:
    d: Dict[Tuple[str, str], int] = {}
    for r in rows:
        d[r.slot] = d.get(r.slot, 0) + 1
    return d


def usable_slots(rows: Sequence[RefRow], *, min_chars: int = 2, min_works: int = 2,
                 min_n: int = 8) -> List[Tuple[str, str]]:
    """有效槽位 = 样本数/不同字符数/不同作品数**都达标** ✓（否则标为不可计算 ✓）。"""
    agg: Dict[Tuple[str, str], Dict[str, set]] = {}
    for r in rows:
        a = agg.setdefault(r.slot, {"c": set(), "w": set(), "n": 0})
        a["c"].add(r.char)
        a["w"].add(r.work)
        a["n"] += 1
    ok = []
    for slot, a in agg.items():
        if a["n"] >= min_n and len(a["c"]) >= min_chars and len(a["w"]) >= min_works:
            ok.append(slot)
    return sorted(ok)
