# 35 — v10a：skel latent 即字条件（从头预训练）+ CPU eval daemon 定稿

> 2026-09-06。上承 32（v9c 联训）、34（硬件瓶颈）。用户决策：**预训练阶段就把
> skel latent 作为字条件注入主干（g 通路），从头预训练，不再依赖 ControlNet 外挂；
> eval 全部走 CPU daemon 单臂 g 模式，不再 base+ctrl 双臂。**

## 1. v10a 设计与实现

**核心语义**：骨架 latent (4,32,32) 经既有 glyph-cond 通路注入主干 token 流——
`glyph_embedder (Conv2d 4→384, patch2)` → 256 个 token → `x += glyph_scale * g_tok`；
CFG 时 **g 两半都给真实骨架**（"字形内容是正条件，CFG 只强化 callig 风格"，`forward_with_cfg` 既有行为）；
训练期 `glyph_drop_prob=0.1` 保持无 g 生成能力。

**已有实现盘点（结论：通路现成，缺 4 处接线）**：
| 组件 | 状态 |
|---|---|
| dit.py `use_glyph_cond` (token-add) / `glyph_inject_layers` (逐层注入) / `glyph_drop_prob` | ✅ 既有（s23/v3b 用过，DiT-2Cond-S/2 v2 架构支持） |
| `forward_with_cfg(g=)` 双半语义 | ✅ 既有 |
| train.py `w_glyph_cond` → dataset 标准字形库 g | ✅ 既有（但那是标准字形库，非实例 skel） |
| **实例 skel latent → g 的接线** | ❌ 本次新增：`--skel-as-glyph-cond`（batch['skel_latent'] → model_kwargs['g']） |
| **eval 侧 g 单臂 + flat json** | ❌ 本次新增：cpu_eval_daemon `--mode pretrain_g` |

`train.py` 补丁共 4 处：argparse（skel_as_glyph_cond / skel_latent_shards_dir /
glyph_drop_prob / glyph_inject_layers）、模型构建（use_glyph_cond OR + 透传）、
训练循环（elif skel_as_glyph_cond → g）、dataset（skel_latent_shards_dir 透传）。

**与 v9c 的关系**：同一假设（实例结构信息应进主干）的两条实现路线——
v9c = ControlNet 注入通路 + 主干解冻联训（v9a 初始化）；
v10a = 主干原生 g 条件 + 从头预训练。v9c 已停（38k 步，best eval 0.7619@22.5k、
skel_iou 0.4229、base 臂 0.5295 > v9a 自身——联训无崩塌且为正收益，ckpts 保留）。

## 2. v10a 配置（最终生效值）

```
global_batch_size=320  compile=true  use_checkpoint=false   ← 三件套必须齐全(见 §4 陷阱)
data_csv=train_fame_clean_v8.csv  latent_shards_dir=final_latents_fame_v8
skel_as_glyph_cond=true  skel_latent_shards_dir=final_skel_latents_fame_1px_v8
glyph_scale_init=0.4  glyph_drop_prob=0.1  glyph_inject_layers=0
freeze_char_table=false  (从头无 DINO index, char 表随机初始化需可训练)
lr=1.5e-4 warmup=3000 max=150k  REPA w=0.1 layer8  auto_eval=false (eval 归 CPU daemon)
```
实测：2.81 steps/s、18.5G/24G、100% util（与 v9a 同档）；前 450 步 loss 0.38 快速下降。

## 3. CPU eval daemon 定稿（`src/eval/cpu_eval_daemon.py` + `cpu_eval_worker.py` + `cpu_sampler.py`）

**双模式**：
- `pretrain_g`（v10a 用）：单臂 g 评测（g=标准字形库 latent，**部署态零样本**——推理时
  任何字的字库骨架就是字条件），两 worker 各半（node0/node0+node1）→ 逐样本列表精确合并 →
  **flat `eval_auto_{step}.json`**（train.py 早停零改动直读）。单臂无 ctrl encoder，
  工作量减半，~9-11 min ≪ 14.9 min 窗口。
- `ctrl_pair`（v8b/v9b/v9c 类）：双臂均衡切分（按实测 0.098/0.190 min/样本加权），
  嵌套 `eval_auto_ctrl_{step}.json`。

**通用机制**：盯 `_active_ckpt_dir.txt`；newest-missing（`.pt.done` 或 mtime>90s 的
`.pt`）；锁文件；LD_PRELOAD jemalloc；taskset 绑物理核（numactl 不可用）；分段
`idx_offset` 全局 PNG 下标；worker 常驻模型/VAE 单次加载。

**验收状态**：v8b 0035000 上 idx_offset 修复后的分段数据已对齐
（base[50:100] 0.5196 / ctrl[50:100] 0.7537，均落在 GPU 全量值 1σ 内；
CPU-vs-GPU 的 ssim 系统偏移 ~+0.008 源于 fp32-vs-bf16 采样路径，**早停只做
CPU 序列内自比较，不受影响**；跨实验 GPU/CPU 数字比较时需记住该偏移）。
完整 n=100 双端验证被训练负载打断，以 v10a 首 ckpt 实战为最终验收。

## 4. 启动陷阱清单（本次踩过的，全部已修/根治）

| 陷阱 | 症状 | 根治 |
|---|---|---|
| train.py 用 **`data_csv`** 不是 `csv` | 落到默认 train.csv → latent KeyError | 配置键修正 |
| **`global_batch_size`** 不是 `batch_size` | batch=16 默认值静默生效 (1.5G/8.5sps) | 配置键修正 |
| **`--use-checkpoint` 默认 True** | 显存减半速度减半 (8G/1.3sps) | **train.py 默认值改 false（根治）** |
| eager 无 compile → OOM@320 (碎片) | rope 60MB 分配失败 | compile=true 必须；expandable_segments 加固 |
| `use_lora` 默认 True 但 lora.py 已删 | 启动即 ValueError | **连根删除**（arg+检查+文件 lora.py/eval_test.py） |
| 从头预训练 + freeze_char_table=true | 冻结随机 char 表 (29% 参数死重) | 配置 false |
| `_sync_work/*.sh` CRLF | `sleep 3\r` invalid | 远程 sed 去 \r（脚本保留在 git，注意 .gitattributes） |

## 5. 当前运行态

- tmux：`v10a`（预训练）、`cpu_eval`（daemon, --mode pretrain_g）、`eval_supervisor`
- v10a：step 450+ / 150k，loss 0.38↓，2.81 sps，18.5G
- 观察项：① 首 ckpt (2500) 的 daemon g-eval 落地时长与 flat json 正确性；
  ② `glyph_scale` 是否漂离 0.4（g 通路梯度生命信号）；③ char 表解冻后的字条件质量
