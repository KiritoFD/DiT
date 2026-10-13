# -*- coding: utf-8 -*-
"""嵌入包（Embedding Pack）—— 把"参考库嵌入"做成**可移植文件** ✓。

为什么要它（用户建议 ✓）
    参考库编码是**贵**的一步（参考库上万张 ✓），而测试输出只有几百张 ✓。
    在远端把参考库**编码一次**、存成 pack 拉回本地 ✓ ⇒ 之后所有评测（新 ckpt / 新 step）
    都能在**本地 GPU**上秒级完成 ✓✓，且**离线、可复现** ✓。

与 `encoder.embed_paths` 的路径缓存**不同** ✗
    · `embed_paths` 的 npz 以 **(ckpt, kind, 路径, mtime, size, mask)** 为键 ⇒ 路径一换就失效 ✗
      （所以远端算的缓存在本地**用不了** ✓）。
    · pack 存的是 **嵌入 + 标签**（writer/script/char/work/path ✓）⇒ 与机器/路径无关 ✓✓。
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

LABEL_KEYS = ("writer", "script", "char", "work", "split", "path")


def save_pack(path: str, emb: np.ndarray, rows: Sequence) -> str:
    """保存 `emb` + 每行的标签 ✓（rows 只需有 LABEL_KEYS 里的属性 ✓）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = {"emb": np.asarray(emb, dtype=np.float32)}
    for k in LABEL_KEYS:
        d[k] = np.array([str(getattr(r, k, "")) for r in rows], dtype=object)
    np.savez_compressed(path, **d)
    return path


def load_pack(path: str) -> Tuple[np.ndarray, List[dict]]:
    """载入 pack ⇒ (emb, rows)，rows 是 dict 列表 ✓（顺序与 emb 一致 ✓）。"""
    with np.load(path, allow_pickle=True) as z:
        emb = np.asarray(z["emb"], dtype=np.float32)
        keys = [k for k in LABEL_KEYS if k in z.files]
        cols = {k: [str(x) for x in z[k]] for k in keys}
    n = len(emb)
    rows = [{k: (cols[k][i] if k in cols and i < len(cols[k]) else "") for k in keys}
            for i in range(n)]
    return emb, rows


def pack_summary(path: str) -> dict:
    """给日志用的一行摘要 ✓（不载入全量数据 ✓）。"""
    with np.load(path, allow_pickle=True) as z:
        n = int(z["emb"].shape[0])
        dim = int(z["emb"].shape[1])
        w = len({str(x) for x in z["writer"]}) if "writer" in z.files else -1
        c = len({str(x) for x in z["char"]}) if "char" in z.files else -1
        k = len({str(x) for x in z["work"]}) if "work" in z.files else -1
    return {"path": path, "n": n, "dim": dim, "n_writer": w, "n_char": c, "n_work": k}
