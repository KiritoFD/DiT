# 922 / 50 — S2 实现说明（三层语义分解 + 局部风格-骨架引导）

> 设计依据：[`20_style_encoding.md`](20_style_encoding.md)（表怎么分级）、
> [`30_injection.md`](30_injection.md)（adaLN vs cross-attn）、[`40_decision.md`](40_decision.md)（决策）。
> 本文只讲**怎么落地的**：改了哪些文件、每个模块的语义、踩到的坑、怎么验证、怎么跑。
>
> 实现日期：2026-09-22 · 状态：**代码完成 + 21 项冒烟全通过** · 尚未开始训练

---

## 0. 一句话

**保留 adaLN 做全局风格主通道；新增"三层语义分解表"与三条廉价的局部通路
（书体 FiLM / 逐位置 FiLM / 风格条件化局部骨架 cross-attn）。**
全部 `zero-init` → **step 0 与旧路径逐位相等**，任何已有 ckpt 可安全 resume 做 A/B。

---

## 1. 交付清单

| 文件 | 改动 |
|---|---|
| `src/model/dit.py` | **新增 4 个模块** + `DiT_2Cond.__init__` 的 12 个开关 + forward / CFG 接线 + zero-init 恢复 |
| `src/model/__init__.py` | 导出 4 个新模块 |
| `src/train/cli.py` | 新增 11 个参数；**修掉一个既存的静默 bug**（见 §5.6） |
| `src/train/train.py` | 把新开关传给模型；hier 模式下注入 `y_callig_raw` / `y_pair` / `y_script` |
| `src/utils/latent_dataset.py` | batch 新增 `y_callig_raw`（书家连续索引）、`y_pair`（pair 索引） |
| `src/eval/inference.py` | `sample_latents` / `sample_latents_self_cond` 支持 `hier_conds`；`make_eval_cache` 产出三个平行 id 列表 |
| `src/eval/in_mem_eval.py` · `batch_eval.py` | 透传 `cache["hier_conds"]` |
| `src/eval/model_io.py` | `build_model_from_args` 补 14 个 S2 字段（**不补会让独立评测静默走旧路径**） |
| `tools/smoke_s2_hier_style.py` | **新增**：21 项 CPU 冒烟（等价性 / 形状 / 梯度 / 掩码 / 局部性） |
| `_sync_work/make_s2_configs.py` | **新增**：生成 7 个单变量消融配置 |
| `src/train/configs/v17_s2_*.json` | **新增**：7 个配置 |

---

## 2. 四个新模块

### 2.1 `StyleHierarchy` — 三层表的"新增两层"

```python
e_style  = e_callig(主效应) + α · E_pair[pair]     # α init 0.1，可学习
e_script = E_script[script]
```

**关键设计决定：书家主效应不新建表，直接复用既有 `y_callig_embedder`。**

既有表已经承载了四套机制，复用后全部零成本可用：

| 机制 | 复用的好处 |
|---|---|
| SupCon 预训练加载（`--callig-emb-pretrained`） | 直接用 `assets/callig_emb_pretrained_50k.pt`（cos 0.020） |
| CFG null 行语义（`label == num_classes`） | uncond 分支自动正确 |
| 冻结（`freeze_table()`） | 与 v13_base 同口径 |
| **few-shot 新增行（`--train-only-new-callig`）** | ★ S2 的卖点"新书家只学 128 维主效应行"**直接可用** |

`StyleHierarchy` 只新增 `E_pair(87×128)` + `E_script(3→12×64)` + `α`，**约 17K 参数**。

**残差为什么能自动回退**：`E_pair` **zero-init** → step 0 时 pair 项恒为 0，
模型退化为"纯主效应"（= v13 配方的超集）。稀疏 pair（15 个 <50 样本，min=1）
梯度微弱 → 残差学不动 → 自动由主效应兜底，**不需要任何硬编码回退逻辑**。

**drop 语义**：callig 被 4-way dropout 丢弃时，pair 项**整项置零**（不是换成 null 行）
→ `e_style = null_callig + 0`，uncond 分支保持纯净。

### 2.2 `ScriptGlyphFiLM` — 通路 B1（书体，结构轴）

```python
γ, β = Linear(E_script[s]).chunk(2, -1)      # zero-init
g_tok = g_tok * (1 + γ) + β * keep
```

**为什么书体不进 adaLN**：`g` 本身就是**按书体渲染**的标准骨架，书体与 g 高度冗余。
独立进 adaLN 会变成"同一个信号喂两遍"（12ch 失败的教训）。正确做法是**调制骨架**。

### 2.3 `SpatialStyleFiLM` — 通路 B2（Phase 1，比 cross-attn 便宜 ~5x）

```python
γ_i, β_i = f( e_cond, g_tok_i, pos_i )       # 每个 token 不同
g_tok_i  = g_tok_i · (1 + γ_i) + β_i · keep
```

与 B1 的区别：B1 的 γ/β **对所有 token 相同**（全局），B2 **逐 token 不同**（局部）。
已经能表达"同一风格下不同局部位置有不同处理"。
成本：256 token × ~106K 参数（D=384, rank=64）≈ **全模型 FLOPs 的 1%**。

**为什么先做它再做 cross-attn**：如果局部性有效但 FiLM 就够了，就不必付 attention 的钱。

### 2.4 `LocalStyleGlyphAdapter` — 通路 B3（Phase 2，局部风格-骨架 cross-attn）

```python
Q   = x 或 g_tok（**带 2D sincos 位置**）
K/V = **局部骨架 token**（**带位置**）        ← 空间证据是真的
style → FiLM 调制 Q/K                        ← 风格决定"看哪里"
out = out_proj(attn)                         # out_proj zero-init
```

**与两个历史方案的本质区别**：

| | 旧 `CalligStyleCrossAttn`（v15b） | 旧 `style_ctx_every_layer`（v15c） | **本 adapter** |
|---|---|---|---|
| K/V 是什么 | K 个**风格 token**（余弦 0.884，假模态） | 骨架 + K 风格 token | **局部骨架**（真实空间证据） |
| 风格怎么用 | 作为被查询对象 | 作为额外 context | **改变注意力分布（调制 Q/K）** |
| 成本 | ~1.02× | **+33%** | ~1.01×（2 层） |
| 实测 | +0.0016（噪声级） | 独立口径**最差** | 待测 |

**`local_ca_q` 两种语义**：

| | Q | 语义 | 代价 |
|---|---|---|---|
| `g`（默认） | `g_tok` | 先在条件侧按风格重组骨架，再注入主干 | 更安全、可回退、只改条件通路 |
| `x` | 主干 `x` | 去噪画布在风格条件下向局部骨架询问证据（更"标准"的 cross-attn） | 改动深入主干，成本略高 |

**窗口注意力**（`--local-ca-window 3/5`）：每个位置只看局部邻域，
更符合"局部书写指引"，也更不容易退化成全局平均。

---

## 3. 数据流（接线位置）

```
batch: y_callig(旧语义) + y_callig_raw(书家45) + y_pair(87) + y_script(0/3/4)
  │
  ├─ StyleHierarchy ──► e_style ──► cond_fusion(cat[e_style, e_glyph_vec]) ──► c ──► adaLN ×12
  │                  └─► e_script ─┐
  │                                │
  └─ g ─► glyph_embedder ─► g_tok ─┴─► ScriptGlyphFiLM(B1) ─► SpatialStyleFiLM(B2)
                                    └─► LocalStyleGlyphAdapter ×N(B3, q=g)
                                    └─► x = x + glyph_scale · g_tok
                                    └─► [q=x 时] 在 block 2 / 6 之后逐层注入
```

**⚠ `y_callig` 与 `y_callig_raw` 必须区分**：
v14/v15 里 `y_callig` 装的是 **pair_id(87)**；hier 模式下主效应表只有 **45** 行。
混用 → **越界或静默串书家**。模型里 `_base_ids = y_callig_raw`，残差用 `y_pair`。

---

## 4. 零初始化等价性（最重要的一条）

所有新模块的"出口"都是 zero-init：

| 模块 | zero-init 的位置 | step 0 的效果 |
|---|---|---|
| `StyleHierarchy` | `E_pair.weight` + `α` 小 | `e_style = e_callig`（残差为 0） |
| `ScriptGlyphFiLM` | `film.weight/bias` | `γ=β=0` → `g_tok` 不变 |
| `SpatialStyleFiLM` | `net[-1]` | 同上 |
| `LocalStyleGlyphAdapter` | `out_proj` + `style_qk[-1]` | `q,k` 不变 + `out=0` → 恒等 |

**实测**：同权重下 `hier_style=0` 与 `hier_style=1`（全通路开）的输出
`max|Δ| = 0.000e+00`（**逐位相等**）。
→ 可以直接 `--resume-full` 已有 ckpt 做单变量 A/B，不需要从头训。

⚠ **梯度的一步延迟**（设计上要知道，不是 bug）：
- `∂L/∂α = ∂L/∂e_style · E_pair`，`E_pair` zero-init → **α 第一步梯度为 0**，
  第二步起才动。
- 同理 `style_qk[-1]` 的梯度要等 `out_proj` 非零。
- 但 `E_pair` / `out_proj` **第一步就有梯度**，所以不是死锁。冒烟 T3e/T3f 已验证。

---

## 5. 踩到的坑（8 条，其中 2 条是既存 bug）

### 5.1 ★ adaLN-Zero 模型在 step 0 **输出恒等于 0**

`block.adaLN_modulation[-1]` 与 `final_layer.*` 全是 zero-init →
每个 block 的 gate=0 → 恒等；`final_layer.linear` = 0 → 输出 0。

**后果**：任何"改变条件应该改变输出"的测试在初始化时**必然得到 Δ=0**，
看起来像"新通路没接上"，其实是初始化特性。**排查了很久。**

**解法**：冒烟测试先 `dezero()`（把这两处推开），且 base/S2 用**同一份权重**。

### 5.2 `register_buffer` 对"已存在的属性名"会抛 KeyError

```python
self.win_mask = None
...
self.register_buffer("win_mask", ...)   # KeyError: attribute already exists
```
与 `ctx_pos_g` 是同一个坑（代码里已有注释警告过）。
**解法**：先 `del self.__dict__["win_mask"]`，或干脆不要预先赋值。

### 5.3 风格 γ/β 的形状广播失配（参考实现里的真实 bug）

q/k 的形状是 `(B, H, N, hd)`，而 γ/β 是 `(B, D)`。
`γ.view(B,1,1,D)` 会与 `hd` 失配（`D = H*hd ≠ hd`）。
**解法**：`γ.view(B, H, hd).unsqueeze(2)` → `(B,H,1,hd)`，按头广播。

### 5.4 加性 β 必须乘 `keep`（否则污染 uncond 分支）

`glyph_drop` 时 `g_tok` 被置零表示"无骨架"。若此时加上非零 β，
被丢弃的样本重新获得条件信号 → **uncond-g 分支被污染**。
乘性 γ 不受影响（`0*(1+γ)=0`），**只有加性 β 需要保护**。
`local_ca` 的输出同理（attention 会从零 token 里聚合出非零值）。

### 5.5 `spatial_film` 依赖 `local_pos`，但后者只在 `local_ca` 开启时注册

配置 `s2d_spfilm`（只开 `spatial_film_rank` 不开 `local_ca_layers`）会
`AttributeError: local_pos`。
**解法**：`spatial_film_rank > 0 或 local_ca_layers > 0` 任一成立就注册。

### 5.6 ★ 既存 bug：`cli.py` 的 `parse_known_args()` 漏传 `argv`

```python
cfg_path = parser.parse_known_args()[0].config   # ← 不传 argv，读的是 sys.argv
```
后果：**只有"真的从命令行启动"时才拿得到 `--config`**；
任何程序化调用 `parse_args(['--config', p])`（测试 / 脚本 / sweep 生成器）
都会静默拿到 `cfg_path="config.json"` → **config 一个字段都不生效**，全部退回代码默认值，**不报任何错**。

（实测踩到：用 `parse_args(['--config', 'v17_s2_s2c_full.json'])` 验证时，
`hier_style` 读到 0、`num_pairs` 读到 0，看起来像"参数没注册"，其实是这个 bug。）

**解法**：`build_parser(argv)` + `parse_known_args(argv)`；`parse_args` 不再重复实现合并逻辑。

### 5.7 `--script-embed-dim` 的历史默认是 `None`

`int(None)` → TypeError。独立评测端会原样透传 None。
**解法**：模型侧 `int(script_embed_dim or 64)`。

### 5.8 `--num-scripts` / `--script-embed-dim` 与既有参数重名

两个参数在 3cond 时代就存在。**不要新增同名参数**，直接复用并更新 help。

---

## 6. 冒烟验证（`tools/smoke_s2_hier_style.py`，CPU，21 项全通过）

```
[PASS] T1a 共享权重全部命中（missing 只应是 S2 新模块）  missing=28 unexpected=0
[PASS] T1b 无 unexpected key（不破坏旧 ckpt 兼容）
[PASS] T1c 非退化前向（de-zero 后条件真的影响输出）  max|out|=4.173e+00
[PASS] T1d zero-init => 与旧路径逐位相等（可安全 resume）  max|Δ|=0.000e+00   ← ★ 最关键
[PASS] T2a forward_with_cfg 跑通
[PASS] T2b forward_with_2axis_cfg 跑通
[PASS] T2c local_ca_q='x' + window=5 跑通
[PASS] T3a E_pair 从 step 0 就有梯度            |g|=2.83e+03
[PASS] T3b script_film 从 step 0 就有梯度        |g|=1.95e+06
[PASS] T3c spatial_film 从 step 0 就有梯度       |g|=2.77e+07
[PASS] T3d local_ca.out_proj 从 step 0 就有梯度  |g|=2.51e+08
[PASS] T3e α 在 E_pair 非零后有梯度（一步延迟，非死锁）
[PASS] T3f style_qk 在 out_proj 非零后有梯度（一步延迟）
[PASS] T4a 换 pair_id 改变输出（pair 残差生效）    max|Δ|=3.26e-01
[PASS] T4b 换 script_id 改变输出（书体 FiLM 生效）  max|Δ|=3.59e-01
[PASS] T4c g=0 改变输出（骨架通路生效）           max|Δ|=2.48e+00
[PASS] T5a drop-g 时 g_tok 恒为 0（β 乘了 keep，不污染 uncond 分支）
[PASS] T5b keep=None 时 g_tok 非零（确认 T5a 不是假阳性）
[PASS] T6a local_ca 逐 token 不同（未退化成全局）      std=8.22e+00
[PASS] T6b spatial_film 逐 token 不同（未退化成全局）  std=7.05e+01
```

另验证（不在脚本内）：

```
S/2 参数量:   v13_base 36.550M  →  s2c_full 36.611M (+61K)  →  s2g_all 38.134M (+1.58M)
旧 ckpt → S2 模型:  missing=6（全是 S2 新模块）  unexpected=0
配置解析:      8 个配置的 14 个字段全部正确读出（v13_base 无回归）
```

---

## 7. 消融阶梯（`_sync_work/make_s2_configs.py`）

**逐级加一样**，不要一次全上 —— 否则涨了不知道是谁的功劳、没涨也不知道是谁的问题。

| 配置 | 加了什么 | 回答什么 |
|---|---|---|
| `v13_base_50k`（已有） | — | 基线：strict 0.5703@155k / ink_ssim 0.3872 |
| `v17_s2_s2a_scriptfilm` | 书体 FiLM（pair 残差**冻结**，参数个数相同） | 书体层有没有用？（与 g 冗余的风险） |
| `v17_s2_s2b_pairres` | pair 交互残差（无书体 FiLM） | 残差分解有没有用？ |
| `v17_s2_s2c_full` | a + b | 组合 |
| `v17_s2_s2d_spfilm` | + 逐位置 FiLM（rank 64） | **局部性**有没有用（便宜） |
| `v17_s2_s2e_lca_g` | + LocalCA ×2（block 2,6，Q=g） | attention 动态聚合是否强于 FiLM |
| `v17_s2_s2f_lca_x` | 同 e 但 Q=x + window=5 | Q 来源 / 窗口 vs 全局 |
| `v17_s2_s2g_all` | 全套 | 叠加上限 |

### 7.1 与 v13_base 的可比性

配置以 `v13_base_50k.json` **逐字段复制**，只改列出的字段：

- 同一份数据（`train_50k_v2.csv`）—— **不换 `_fixed`**，否则与 v13_base 不可比
- 同一 schedule（`max_steps 250000` / warmup 3000 / cosine / min_lr_ratio 0.1）
  → **任意 step 都能同口径对比**，跑到 ~160k 即可判读
- 同样冻结书家表 + 同一份 SupCon 预训练表
- `glyph_drop_prob` 保持 **0.0**（不引入第二个变量）

⚠ 要换清洗后的数据：`python _sync_work/make_s2_configs.py --fixed-data`，
但**必须另跑一条 `_fixed` 基线**，否则 v13_base 不再是合法对照。

### 7.2 跑法

```bash
ssh 4090 && cd /root/Workspace/xy/DiT && export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

$PY tools/smoke_s2_hier_style.py                       # 先冒烟（1 分钟内，可本地跑）

# 串行跑阶梯（每个 250k 上限，~160k 即可判读；4.1 step/s @ batch 360 ≈ 11h/条）
for c in s2a_scriptfilm s2b_pairres s2c_full s2d_spfilm s2e_lca_g; do
  $PY src/train/train.py --config src/train/configs/v17_s2_$c.json
done
```

**起训后必看的三行日志**（少了就不要判读）：

```
[channels] latent=4 aux 组=0 in_channels=4 image_channels(CFG 作用域)=4
[hier-style] ...                      ← 应打印 num_pairs / num_scripts / pair_residual
[style-anchor] ...                    ← 若开了 anchor
```

（`[hier-style]` 这行**还没加** —— 见 §8 TODO。）

---

## 8. 判据（**不要用 strict ssim**）

| 判据 | 基线 | 目标 | 备注 |
|---|---|---|---|
| ★ **风格可分性探针 T2** | 待测（随机 1.1%/87 类） | **> 20%** | 生成图提 DINO → 训分类器测 top-1；跨 ckpt 可比、不需同 cfg |
| ★ **few-shot 新书家 Δssim** | +0.010~+0.038 | **≥ +0.05**（冻结主干） | S2 的卖点，用 `fs6_paired.py` 做配对检验 |
| `ratio_style`（cfg=1.0，≥100 对） | 1.21–2.48 | **≥ 2.5** | 现在口径混，需重测 |
| `ink_ssim`（strict 249） | 0.3898 | 不能变差 | 兜底 |
| ~~strict ssim~~ | 0.573 | — | ✗ 已饱和，效应量 <0.01 |

> **如果 S2 在 strict ssim 上零增益、但 T2 / few-shot 明显涨 —— 这就是成功。**

---

## 9. 还没做的（TODO）

| # | 事项 | 优先级 |
|---|---|---|
| 1 | **加 `[hier-style]` 启动日志**（打印 num_pairs / pair_residual / 各通路开关与参数量） | 高（防"配置没生效"静默事故） |
| 2 | **训练前先跑 D1**（测 `‖y_emb‖/‖t_emb‖`）—— 若 ≪1 说明风格被 timestep 淹没，S2 大概率也白搭 | **最高** |
| 3 | **P0 口径统一**：v15c 的 0.126 差异、cfg=1.0 重测、`v15b_supcon` 终值 | 高 |
| 4 | T2 探针脚本（生成图 → DINO → 线性分类器） | 高（判据依赖它） |
| 5 | few-shot 在 S2 上的重跑（`--init-new-callig row_pt` 对 128 维主效应行） | 中 |
| 6 | `E_pair` 的层级 SupCon 预训练（复用 `pretrain_callig_script_emb.py` 的层级正对矩阵） | 低（zero-init 起步够用） |
| 7 | attention 热力图可视化（看是否集中在起笔/收笔/转折/波磔） | 中（判据之一） |
| 8 | `--local-ca-window` 的 3 vs 5 vs 全局对照 | 低 |
