# -*- coding: utf-8 -*-
"""src/eval/legacy/ —— **已退役的评测实现**（只读归档 ✓ 不在运行时路径上 ✗）

2026-10-14 [重构 PR3] 建立本目录，用于按审查 §5 的**正确删除流程**逐步退役历史评测入口:

    静态引用扫描 -> 历史 config 引用扫描 -> 训练/评测入口动态 import 测试
      -> 确认 checkpoint 兼容 -> 迁移外部工具调用 -> 移出运行时路径
      -> **一个发布周期后删除** ✓

★ 归档不等于删除 ✓: 文件保留在本目录（git 历史与内容都在 ✓），
  历史 checkpoint 对应的配置与源码**不会因为清理而失去可复现性** ✓（审查要求 ✓）。

★ 为什么放在 `legacy/` 而不是直接删 ✗:
  · 它们仍是"某次历史实验是怎么算出来的"的唯一证据 ✓
  · 但**不应**再被新代码 import ✗（新代码请用 `src/eval/sampler.py` 的单一采样语义 ✓）

已归档:
  · `gpu_batch_eval.py`     —— 独占 GPU 批量评测（**自建 diffusion + 自跑采样循环** ✗）
  · `gpu_batch_eval_v2.py`  —— 上一版 GPU 批量评测（同上 ✗）

  退役依据（可复核 ✓）:
    ① 静态扫描: 全仓库真实 `import` 引用 = **0** ✗（仅自身文档串 + 一条注释 ✓）
    ② 配置扫描: `*.json` 引用 = **0** ✗
    ③ 动态 import: 可导入但**无人调用** ✓
    ④ 替代实现: `src/eval/sampler.py` 的 `sample_latents` ✓（**已被验收** ✓）
       —— 它已被 `in_mem_eval` / `batch_eval` / `eval_ctrl_ckpt` 采用 ✓
    ⑤ 外部工具调用: **无** ✓ ⇒ 无需迁移 ✓

★ 判别"自实现"的**正确**标准 ✗（本目录建立时的一处措辞已订正 ✓）:
    共享采样器把 `diffusion.ddim_sample_loop(...)` 作为**统一接口** ✓
    （`inference.py` 明确写着"采样统一走 `diffusion.ddim_sample_loop(..., clip_denoised=False)`" ✓
      —— `sampler.py` **自己**也调用它 ✓）
    ⇒ `ddim_sample_loop` / `create_diffusion` **不是**遗留标记 ✗
    ⇒ 真正的判别是: **该入口是否自建 diffusion 并自跑采样循环** ✗
       （应改为调用共享 `sample_latents` ✓）
    仍属此类（**尚未归档** ✗，待迁移其调用方后处理 ✓）:
      `in_process_eval.py` / `eval_auto.py` / `eval_full_3cond.py` / `auto_eval_ctrl.py`
      —— 注意 `auto_eval_ctrl.py` 等是**以脚本方式被调用**的活跃入口 ✓
         （0 import ≠ 死代码 ✗ ⇒ 必须先迁移外部调用 ✓ 再归档 ✓）
"""
