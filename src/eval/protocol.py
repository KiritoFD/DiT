# -*- coding: utf-8 -*-
"""protocol.py — **评测的实验身份（eval_id）**（2026-10-13 [审查 §4.3]）

审查原文:
  "当前 in_mem_eval.py 仍以 (step, set) 为主要完成判据。这不足以区分同一步数下的
   不同模型、CFG、TTF 或采样参数。"
  "每轮先创建不可变协议 … 序列化为规范 JSON 后计算 SHA256，作为 eval_id。"
  "DONE.json 只有在所有样本、评分文件和哈希校验成功之后才写入。"

本模块把这条要求做成**可复用、可测试**的一小块（不依赖任何评测器实现 ✓）:
  · `EvalRequest`  : 冻结协议（审查给的字段 ✓ 一个不多一个不少）
  · `eval_id`      : 规范 JSON 的 sha256 ⇒ **任一字段不同 ⇒ 不同 id** ✓✓
  · `is_complete`  : 不仅看 `DONE.json` 存在 ✓, 还**核对其中记录的哈希仍与实际一致** ✓✓
                     ⇒ 防"残留图/错图复用"（本项目历史上的真实事故 ✗）
  · `sha256_file` / `sha256_tree` / `git_revision` : 身份素材的来源 ✓

设计约束（刻意的）:
  · **零依赖**于 in_mem_eval / inference —— 谁都能调, 也便于单测 ✓
  · 缺文件一律**抛错而不是猜** ✗（"缺失就停止" 是审查反复强调的原则 ✓）
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

__all__ = ["EvalRequest", "eval_id", "artifact_dir", "protocol_path", "done_path",
           "write_protocol", "write_done", "is_complete", "load_done",
           "sha256_file", "sha256_tree", "git_revision", "SCHEMA_VERSION",
           "png_inventory"]


def png_inventory(png_dir: str) -> Dict[str, str]:
    """公开封装 ✓（测试与外部审计用 ✓ —— 见 `_png_inventory` ✓）。"""
    return _png_inventory(png_dir)


SCHEMA_VERSION = 1


# ────────────────────────────── 身份素材 ──────────────────────────────
def sha256_file(path: str) -> str:
    """单文件 sha256（缺失即报错 ✗ —— 不返回空串冒充 ✓）"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: str, exts: Tuple[str, ...] = (".csv", ".json")) -> str:
    """目录内容指纹: **按相对路径排序**后逐个哈希再汇总 ✓

    用于 dataset / std-soft store 这类"多文件但有确定性内容"的身份 ✓。
    只统计给定后缀（默认 csv/json ✓）, 避免把随机产物混进身份 ✗。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"sha256_tree: 目录不存在 {root!r} ✗")
    h = hashlib.sha256()
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if exts and not fn.endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            h.update(rel.encode() + b"|" + str(os.path.getsize(p)).encode() + b"|")
            h.update(bytes.fromhex(sha256_file(p)))
            n += 1
    h.update(b"|n=%d" % n)
    return h.hexdigest()


def git_revision(short: bool = True) -> str:
    """代码版本（拿不到就返回 'unknown' **并让调用方知道** ✓ —— 不假装有值 ✗）"""
    try:
        # ⚠ 原来写成 ["git","rev-parse", "--short" if short else "HEAD"] ✗ —— 条件表达式
        #   落在列表元素上, 于是命令变成 git rev-parse --short（缺 HEAD ✗）⇒ 永远返回 unknown ✓
        _args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        r = subprocess.run(_args, capture_output=True, text=True, timeout=15)
        rev = (r.stdout or "").strip()
        if not rev:
            return "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=15).stdout.strip())
        return rev + ("-dirty" if dirty else "")
    except Exception:                                             # noqa: BLE001
        return "unknown"


# ────────────────────────────── 冻结协议 ──────────────────────────────
@dataclass(frozen=True)
class EvalRequest:
    """一次评测的**不可变**协议 ✓（字段 = 审查给的清单 ✓）

    ⚠ 新增字段时: 必须同时升 `SCHEMA_VERSION` ✓ —— 否则旧 DONE.json 会被误判为可比 ✗。
    """
    checkpoint_sha256: str
    vae_sha256: str
    vae_scale: float
    vae_shift: float
    sampler: str
    num_steps: int
    cfg: float
    gamma: float
    seed: int
    ema: bool
    code_revision: str
    dataset_sha256: str = ""
    conditions_sha256: str = ""
    sets_spec: str = ""                    # 评测集定义（如 "eval200fix:...:187,seen:...:20"）
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # ---- 规范化序列化（**排序** ⇒ 同内容必同 id ✓）----
    def canonical(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str)

    def eval_id(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        d = asdict(self)
        d["eval_id"] = self.eval_id()
        return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)

    def validate(self) -> "EvalRequest":
        """身份完整性: 关键哈希不许为空 ✗（空串会让不同实验撞成同一个 id ✗✗）"""
        for k in ("checkpoint_sha256", "vae_sha256"):
            v = getattr(self, k)
            if not v or len(v) < 16:
                raise ValueError(f"EvalRequest.{k} 缺失或过短（{v!r}）—— "
                                 f"身份必须可溯源, 不许用空值冒充 ✗")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version={self.schema_version} 与本模块 "
                             f"{SCHEMA_VERSION} 不一致 ⇒ 旧结果**不可比** ✗")
        return self


def eval_id(req: EvalRequest) -> str:
    return req.validate().eval_id()


# ────────────────────────────── 产物目录与完成判定 ──────────────────────────────
def artifact_dir(root: str, eid: Optional[str] = None) -> str:
    """`<root>/evaluations/<eval_id>/`（审查建议的目录布局 ✓）"""
    return os.path.join(root, "evaluations", eid) if eid else os.path.join(root, "evaluations")


def protocol_path(eval_dir: str) -> str:
    return os.path.join(eval_dir, "protocol.json")


def done_path(eval_dir: str) -> str:
    return os.path.join(eval_dir, "DONE.json")


def write_protocol(req: EvalRequest, eval_dir: str) -> str:
    os.makedirs(eval_dir, exist_ok=True)
    p = protocol_path(eval_dir)
    with open(p, "w", encoding="utf-8") as f:
        f.write(req.validate().to_json())
    return p


PNG_GLOB_PREFIX = "g"


def _png_inventory(png_dir: str) -> Dict[str, str]:
    """收集目录下全部 `g<数字>.png` ⇒ {文件名: sha256} ✓（排序保证确定性 ✓）。"""
    import re
    out: Dict[str, str] = {}
    if not os.path.isdir(png_dir):
        return out
    pat = re.compile(r"^g\d+\.png$")
    for fn in sorted(os.listdir(png_dir)):
        if pat.match(fn):
            out[fn] = sha256_file(os.path.join(png_dir, fn))
    return out


def write_done(req: EvalRequest, eval_dir: str, n_samples: int,
               metrics_path: Optional[str] = None,
               png_dir: Optional[str] = None,
               manifest_names: Tuple[str, ...] = ("samples.csv", "samples.jsonl")) -> str:
    """**只有跑完、且哈希自校验通过**才写 DONE ✓（审查: "只有在所有样本、评分文件和
    哈希校验成功之后才写入" ✓✓）

    ★ 2026-10-14 [重构 P0-1] 修复审查已复现的缺陷 ✗:
        旧实现只哈希"碰巧存在"的文件 ⇒ 无样本也能写出 DONE ✗✗
        且 is_complete 只核对已登记产物 ⇒ 伪造的空 DONE 也判"完整" ✗✗
    新语义（**缺任一必需产物 ⇒ 抛错, 绝不写 DONE** ✗）:
      ① 样本 manifest（samples.csv / samples.jsonl 任一 ✓）必须存在 ⇒ 否则 raise ✗
      ② PNG 目录中 `g<数字>.png` 数量必须**恰好等于 n_samples** ⇒ 否则 raise ✗
      ③ DONE 记录全部 PNG 的 sha256 + manifest/metrics 的 sha256 ✓
      ④ DONE **原子写入**（tmp + os.replace ✓ —— 半写的 DONE 不会再被当成完整 ✓）
    """
    req = req.validate()
    arts: Dict[str, str] = {}

    # ① manifest 必须存在（任一命名 ✓ —— in_mem_eval 实际写 samples.jsonl ✓）
    hit = next((m for m in manifest_names
                if os.path.isfile(os.path.join(eval_dir, m))), None)
    if hit is None:
        raise RuntimeError(
            f"write_done: 样本 manifest 缺失（找过 {manifest_names} @ {eval_dir}）✗ —— "
            f"采样未完成或产物丢失 ⇒ **拒绝写 DONE** ✗")
    arts[hit] = sha256_file(os.path.join(eval_dir, hit))

    # ② PNG 清点（必须在写 DONE 前就**恰好等于 n_samples** ✓）
    pdir = png_dir or eval_dir
    pngs = _png_inventory(pdir)
    if len(pngs) != int(n_samples):
        raise RuntimeError(
            f"write_done: PNG 数 {len(pngs)} ≠ n_samples={n_samples}（目录 {pdir}）✗ —— "
            f"缺图/多图/错图 ⇒ **拒绝写 DONE** ✗")
    for fn, h in pngs.items():
        arts["png:" + fn] = h

    if metrics_path and os.path.isfile(metrics_path):
        arts[os.path.basename(metrics_path)] = sha256_file(metrics_path)
    # 约定命名的产物: 存在即登记 ✓（**保留旧约定** ✓ —— 但**不再**"存在就写 DONE" ✗：
    #   必须的 manifest/PNG 已在上面**强制校验** ✓ 这里只是把可选产物也纳入哈希链 ✓）
    for name in ("metrics.json", "manifest.json", "encoder_audit.json"):
        p = os.path.join(eval_dir, name)
        if os.path.isfile(p):
            arts[name] = sha256_file(p)
    done = {
        "eval_id": req.eval_id(),
        "protocol_sha256": sha256_file(protocol_path(eval_dir)),
        "schema_version": req.schema_version,
        "n_samples": int(n_samples),
        "n_png": len(pngs),
        "artifacts": arts,
    }
    p = done_path(eval_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)                                     # ★ 原子替换 ✓
    return p


def load_done(eval_dir: str) -> Optional[Dict[str, Any]]:
    p = done_path(eval_dir)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return None


def is_complete(req: EvalRequest, eval_dir: str,
                png_dir: Optional[str] = None) -> Tuple[bool, str]:
    """完成判定 —— **不只看 DONE 存在** ✓（这正是历史事故的根因 ✗）

    ★ 2026-10-14 [重构 P0-1] 修复审查已复现的缺陷 ✗:
        旧实现只核对 DONE **已登记**的产物 ⇒ 未登记的缺失/伪造全放行 ✗✗
        （实测: 无 PNG 无指标也能返回 True ✗）
    新增三道**从磁盘重算**的检查（不信任 DONE 的自我申报 ✓）:
      ⑤ manifest 文件必须在盘上 ✓
      ⑥ 盘上实际 PNG 集合（数量+文件名集合）必须等于 DONE 记录 ✓
      ⑦ 每张 PNG 的 sha256 必须与 DONE 记录一致 ✓
        ⇒ 删一张/换一张/加一张 PNG ⇒ 判未完成 ✗✗（专家验收标准 ✓）
    返回 (是否可复用, 原因)。只有下面**全部**成立才算完成:
      ① DONE.json 存在且可解析
      ② 其中的 `eval_id` **等于本请求**的 id（同 step 不同模型/CFG/TTF ⇒ 不等 ✓✓）
      ③ `protocol.json` 的哈希与 DONE 里记录的一致（协议没被改过 ✓）
      ④ DONE 里记录的每个产物**文件仍在且哈希一致**（防残留/错图 ✗）
      ⑤⑥⑦ 见上 ✓
    """
    eid = req.validate().eval_id()
    d = load_done(eval_dir)
    if d is None:
        return False, "无 DONE.json"
    if d.get("eval_id") != eid:
        return False, f"DONE 的 eval_id={str(d.get('eval_id'))[:12]} ≠ 本次 {eid[:12]}（协议不同 ⇒ 必须重跑 ✓）"
    if d.get("schema_version") != SCHEMA_VERSION:
        return False, f"schema_version={d.get('schema_version')} ≠ {SCHEMA_VERSION}"
    pp = protocol_path(eval_dir)
    if not os.path.isfile(pp):
        return False, "protocol.json 丢失"
    if sha256_file(pp) != d.get("protocol_sha256"):
        return False, "protocol.json 已被修改（哈希不符 ✗）"
    for name, h in (d.get("artifacts") or {}).items():
        if name.startswith("png:"):
            continue                    # ★ PNG 由下面的**磁盘重算**统一校验 ✓（P0-1 ✓）
        p = os.path.join(eval_dir, name)
        if not os.path.isfile(p):
            return False, f"产物缺失: {name}"
        if sha256_file(p) != h:
            return False, f"产物哈希不符: {name}（被改动/覆盖 ✗）"
    # ⑤ manifest 必须在盘上 ✓（旧 DONE 若没登记 manifest ⇒ 在此显式失败 ✗）
    arts = d.get("artifacts") or {}
    if not any(k in arts for k in ("samples.csv", "samples.jsonl")):
        return False, "DONE 未登记样本 manifest ⇒ 旧协议产物, 不可复用 ✗"
    # ⑥⑦ PNG 集合**从磁盘重算** ✓（不信任 DONE 的自我申报 ✓）
    pdir = png_dir or eval_dir
    now_pngs = _png_inventory(pdir)
    rec_pngs = {k[len("png:"):]: v for k, v in arts.items() if k.startswith("png:")}
    n_exp = int(d.get("n_samples", -1))
    if len(now_pngs) != n_exp or len(rec_pngs) != n_exp:
        return False, (f"PNG 数不符: 盘上 {len(now_pngs)} / DONE {len(rec_pngs)} / "
                       f"应有 {n_exp} ✗")
    if set(now_pngs) != set(rec_pngs):
        missing = sorted(set(rec_pngs) - set(now_pngs))[:3]
        extra = sorted(set(now_pngs) - set(rec_pngs))[:3]
        return False, f"PNG 文件集合不符 ✗ 缺={missing} 多={extra}"
    for fn in sorted(now_pngs):
        if now_pngs[fn] != rec_pngs[fn]:
            return False, f"PNG 哈希不符: {fn}（被替换/重生成 ✗）"
    return True, f"完整可复用 ✓（{d.get('n_samples')} 个样本, eval_id={eid[:12]}）"
