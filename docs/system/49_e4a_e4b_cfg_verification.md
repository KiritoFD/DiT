# 49. E4a/E4b 实测：CFG 口径修正的边界 + 历史曲线重测（2026-09-10）

前置: 48（GPU 消融 + CFG 发现）
工具: `tools/eval/e4a_cfg_sweep.py`（strict n=237）、`tools/eval/e4b_hist_cfg.py`（seen n=10）

---

## 1. E4a：strict 集（n=237）CFG 复核 — 修正 doc48 §2 的适用范围

ckpt: c41x @212.5k（与 doc48 同一 ckpt），strict = `assets/eval_fame3_strict_clean_v9.csv`，
Heun50，skel hit 237/237：

| cfg | ssim | skel_iou |
|---:|---:|---:|
| 0.7 | 0.5481 | 0.0158 |
| **1.3** | **0.5529** | 0.0171 |
| 1.5 | 0.5527 | 0.0179 |
| **2.0** | 0.5501 | **0.0182** |
| 2.5 | 0.5484 | 0.0181 |
| 3.0 | 0.5459 | 0.0179 |

**结论：**

1. **CFG 修正主要作用于 seen（记忆化）指标**：doc48 的 skel_iou 0.223→0.352（+58%）
   是 seen 集的发现；**strict 新字集上 CFG 只给 +0.0024 iou / +0.005 ssim**
   （精确 1px 骨架 IoU 始终 ~0.016-0.018，近零）。
2. **strict 口径的历史结论不受影响**（0.7 vs 2.0 差异在噪声量级）；
   **seen 口径的历史曲线需要修正**（E4b）。
3. ssim 在 strict 上的最优 ≈1.3，iou 最优 ≈2.0，曲线平坦；**新 config 统一 `eval_cfg=2.0`**。

---

## 2. E4b：c41x 历史 seen 曲线重测（cfg 0.7 vs 2.0）

n=10，与历史 `eval_auto_*.json` 同集同 seed；**cfg=0.7 一列精确复现 doc48 数值**（口径验证 ✓）：

| step | cfg0.7 ssim | cfg0.7 iou | cfg2.0 ssim | cfg2.0 iou |
|---:|---:|---:|---:|---:|
| 105k | 0.6456 | 0.0602 | 0.7070 | 0.1112 |
| 130k | 0.7002 | 0.0925 | 0.7607 | 0.1871 |
| 155k | 0.7448 | 0.1489 | 0.7835 | 0.2711 |
| 180k | 0.7642 | 0.1865 | 0.8206 | 0.3255 |
| 205k | 0.7798 | 0.2248 | 0.8232 | 0.3430 |
| 212.5k | 0.7838 | 0.2227 | **0.8261** | **0.3517** |

**关键修正：**

- cfg=0.7 口径下 180k 后 ssim 平台（0.764→0.784）在 **cfg=2.0 下并不平台**：
  0.8206→0.8232→0.8261 仍在缓升，iou 0.3255→0.3430→0.3517 也在升。
- **"c41x 容量已达平台"的结论建立在错误口径上**，需在新口径下重新审视
  （间接支持 scratch 从头训 400k 的决定）。
- sty16@15k（容量扫描对照）：cfg0.7 0.4949/0.0155 → cfg2.0 0.5163/0.0135。
  （与 c41x 不同 step 不可直接比；容量扫描结论以 48 号为准。）

---

## 3. 意外事件：scratch 训练被杀（非崩溃）

- 13:37 启动的 c41x_scratch（from scratch）在 **14:34 突然消失**（tmux 会话 + 所有
  进程，无 Traceback、无 dmesg OOM、机器未重启、日志停在正常训练行）。
  最可能原因：并行会话为释放 GPU 运行 doc48 消融而 `pkill`。
- 已于 17:25 从 0012500.pt resume（fresh scheduler，绝对步数 cosine，
  LR 4.99e-5 接续，missing=0/unexpected=0）。

---

## 4. 下一阶段实验（config 已就绪，待排期）

基座 = c41x_scratch（style_attn + 增强 e + 5e-5 cosine + 400k），30k 步早期对照，
**eval_cfg=2.0**：

| config | 唯一变量 | 问题 |
|---|---|---|
| `..._scratch_l0.json` | `glyph_inject_layers=0` | 12 层 xattn 是否过度？(输入 token-add 已撑主要通路) |
| `..._scratch_l4.json` | `glyph_inject_layers=4` | 4 层够不够 |
| `..._scratch_norepa.json` | `w_repa=0` | REPA 0.03 从未被验证 |
| `..._scratch_nodrop.json` | `cond_drop_*=0` | 与 CFG 口径的联动（cfg>1 需要 uncond；闭集部署对照） |

排期注意：GPU 单卡，主训练（400k，~27h）占用中；消融串行插入或等主训练后。

---

## 5. 本轮结论清单

1. CFG 峰值 **seen≈2.0 / strict≈1.3-2.0**；历史 seen 指标系统性低估 +0.04 ssim / +50% iou。
2. strict 泛化骨架遵循**不受 CFG 拯救**（精确 IoU ~0.016）：泛化遵循是模型能力问题，
   不是评测口径问题 —— 这提高了 style_attn / 增强 / 更长训练的价值权重。
3. c41x 在新口径下**未确认平台** → scratch 长训（400k）路线得到支持。
4. c41x_scratch 已 resume（12.5k → 400k）。
5. 评测守护修复（部署于本日）：
   - **F4**：`*.cpu_eval.lock` 残留 >45min 自动清理重评；run_pair 启动时清理
     `.part_*.json` 残留（否则失败评测被误判成功、落盘错误指标）。
     实测清理了 13:34 被杀留下的 `0010000.cpu_eval.lock`。
   - **F5**：worker/批量评测器 unexpected keys 崩溃前打印 missing/unexpected 明细。

