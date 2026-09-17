# 70. 代码审查：冗余 / 性能 / 正确性

> 日期：2026-09-17 ｜ 范围：`src/model`、`src/train`、`src/utils`、`src/loss`、`src/eval`、`tools`
> 方式：**纯代码阅读 + 定点微基准**（未做系统性 profile）
> 相关：[67](67_12ch_xattn_audit.md)（12ch/xattn 实现审计）、[68](68_cfg_impl_diversity_data_overview.md)（CFG）

---

## 0. 结论速览

| 档 | 项 | 影响 |
|---|---|---|
| **P0 正确性** | 8+ 份 `strict=False` 的模型重建 | 形状不匹配时**静默用随机权重**，指标全废 |
| **P0 正确性** | `img_id` 靠文件名正则提取 | 换命名/非数字文件名 → 崩或**静默取错 latent** |
| **P0 正确性** | `in_mem_eval` 多处 `except Exception` 吞异常 | 失败只留空列，**指标静默失真** |
| **P1 性能** | `heun_batch=True`（默认） | **实测慢 23.6%**（B=8；大 batch 应更差） |
| **P1 性能** | 非 preload 路径每样本 `np.load` 整个 shard | 应加 `mmap_mode="r"` |
| **P1 性能** | preload 10 GB 图像，但 REPA 有 DINO 缓存 | 全命中时纯浪费 RAM + 解码时间 |
| **P2 冗余** | `src/eval/` 9 个死文件 | 可删 |
| **P2 冗余** | SSIM 14 份 / VAE 加载 11 份 / skel_iou 8 份 / LPIPS 6 份 | 抽公共模块 |
| **P2 冗余** | `dit.py` 2 路 CFG 白做一次 `cat` | 省 2× 分配 |
| **P3 精简** | `train.py` main() 1630 行 + 150+ 处 `getattr` 默认值互相不一致 | 归一为 dataclass |

---

## 1. 正确性风险（最高优先）

### 1.1 ★ 8+ 份 `strict=False` 的模型重建 —— 最危险的一类

从 ckpt 的 args 重建模型，有 **9 处独立实现**，只有 1 处有护栏：

| 实现 | 加载方式 | 字段完整性 |
|---|---|---|
| `tools/cfg_sweep.py:29 build_model_from_args` | **`strict=True`** ✓ | 最全（逐一对齐 train.py） |
| `tools/eval_diversity.py:133 build_model` | 已废弃（现 exec cfg_sweep），**旧函数仍留着** | 死代码 |
| `src/eval/batch_eval.py:104` | `strict=False` | ✗ 硬编码 `use_glyph_cond=True`，**缺 `glyph_vec_cond`/`glyph_vec_dim`/`glyph_embedder_sep`** |
| `src/eval/cpu_eval_worker.py:167` | `strict=False` | ✗ 缺 `style_token_n` |
| `src/eval/gpu_ablate_eval.py:83` | `strict=False` | ✗ 注释称"含 style_token_n"实则没有 |
| `tools/rerun_eval_wz.py:48` | `strict=False` | ? |
| `tools/eval_sweep_12ch.py:49` | `strict=False` | ? |
| `src/eval/auto_eval_cpu.py:76` | `strict=False` | ? |
| `src/eval/auto_eval_gpu.py:47` | `strict=False` | 已过期（硬编码 `learn_sigma=True`） |

**为什么危险**：v12 是 `factorized_cat` + `glyph_vec_cond=True`，
`cond_fusion.0.weight` 在 ckpt 里是 **`[256]`**（callig128 + glyph_vec128），
而漏了 `glyph_vec_cond` 的重建给 **`[128]`** → `strict=False` 下**静默跳过**，
`cond_fusion` 保持随机初始化 → **测出来的指标毫无意义但看着正常**。

**我们已经在 `tools/eval_diversity.py` 上真实踩过一次**（见 [68 §2.2 D1]）。

**修法**：全部统一到 `cfg_sweep.build_model_from_args`（它已 `strict=True`），
或把该函数提升到 `src/eval/model_io.py` 作为唯一入口。

### 1.2 ★ `img_id` 靠文件名正则提取

```python
ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in self.samples]
```

（`src/utils/latent_dataset.py:192`，latent/skel/aux 的 shard 查表全依赖它）

实测：

| 路径 | 提取结果 |
|---|---|
| `data/50k/imgs/051419.png` | `051419` ✓ |
| `data/fame-kxl-tj-px60/imgs/77.png` | `77` ✓ |
| `data/hcsu_kxl/imgs/bei/祝允明-楷/克.png` | **`None` → AttributeError 崩** |
| `assets/a1/b23/c456.png` | `456`（恰好对，但纯属运气） |

**问题本质**：**img_id 是"文件名约定"而不是 CSV 里的显式字段**。
CSV 里没有 `img_id` 列 → 任何重命名/换命名规则都会让 shard 查表**静默错位**。

**修法**：CSV 增加显式 `img_id` 列并断言非空；正则只作为兼容回退。
（新的 `assets/train_50k.csv` 已有 `old_50k_id` 列可直接用）

### 1.3 `in_mem_eval` 的静默失效面

| 位置 | 行为 | 风险 |
|---|---|---|
| `:67`、`:92` LPIPS 两处 `except Exception` | 吞掉异常，只留空列 | **指标静默缺失** |
| `_get_callig_map:106` 路径不存在返回 `None` | 书家 id **不映射** | 条件静默错位 |
| `inference.py:538` shards 缺失只 print WARNING | 继续跑 `g=ZERO` | 字条件静默失效 |
| `save_input_g:470`、poster `:504` try/except | 吞掉 | 产物缺失无感知 |

**建议**：区分"可降级"与"必须失败"。指标计算失败应当 **raise**；
shards 缺失应当在 **启动时** fail-fast（训练前就知道，而不是跑完 50 步才发现）。

### 1.4 其它正确性隐患

- **`decode` 精度**：`in_mem_eval.py:437` 用 bf16 autocast decode，
  而 `in_process_eval.py:259` 明确要求 **fp32 force_upcast** → 两者口径不一致，存在精度隐患
- **`heun_batch` 的重算**：第 2 次 2B forward 里的 `v1_` 是 `f(x, t_i)` 的**重算**
  （第 1 次已算过）。语义等价（eval 无随机性），但**多跑 50% 的 forward**（见 §2.1）
- **配置键静默失效**：`train.py` 里大量键 `getattr` 读取但**未注册 argparse**
  → config JSON 写了也**被丢弃**。已知：`repa_warmup`(745)、`callig_proj_mode`(295)、
  `callig_scale_init`(296)、`glyph_noise_scale/prob`、`glyph_patch_drop`(1296-1298，
  **整段条件增强是死代码**)、`use_glyph_cond`(1182)、`early_stop_min_delta_lpips`(1055)。
  另有 config 里已出现却未注册的：`cond_drop_all`/`cond_drop_one`（应为 `*_prob`）、
  `scheduler`（应为 `lr_schedule`）、`eval_every`、`use_lora`/`lora_r` 等。
  （注：`repa_layers` 现已注册 `train.py:2191`，v12 config 注释里"未注册"的说法已过期）

---

## 2. 性能

### 2.1 ★ `heun_batch=True` 实测慢 23.6%（默认值，应改）

**机制**：`_v` 调的是 CFG wrapper，每次评估都被 CFG 再翻倍：

| | `heun_batch=True` | `heun_batch=False` |
|---|---|---|
| 第 1 次调用 | B → CFG → **2B** | B → CFG → 2B |
| 第 2 次调用 | 2B → CFG → **4B** | B → CFG → 2B |
| **每步合计** | **6B 行** | **4B 行** |

**微基准**（S/2 36.55M，B=8，heun 20 步，fp16 autocast，取 3 次均值）：

```
heun_batch=True      45.16 ms/步
heun_batch=False     36.53 ms/步     → **慢 23.6%**
```

⚠ 这是在 **B=8** 下测的 —— 小 batch 本来对"拼批"有利。真实 eval 用 `in_mem_eval_batch`
（默认 240），kernel launch 占比更低，**差距应更接近理论值 50%**。

**佐证**：`src/eval/cpu_sampler.py:8` 的注释自己写着
"Heun 两 stage 分开评估 —— `FlowMatching.heun_batch=True` 每步处理 **6B 行**"，
即 **CPU 侧已经主动避开了它，GPU 侧没有**。

**修法**：`FlowMatching.__init__` 的 `heun_batch` 默认改 `False`，
或至少在 `inference.py` 构造 diffusion 时显式传 `heun_batch=False`。

### 2.2 非 preload 路径每样本 `np.load` 整个 shard

`latent_dataset._get_latent:167-179`：

```python
with np.load(shard_path) as shard:      # 无 mmap_mode -> 解压整个 shard
    latent = np.array(shard["latents"][offset], copy=True)
```

取**一行**却解压**整个 shard**。skel / inst / aux 各再来一次。
`preload=True` 时走不到这条路，但一旦关掉 preload（或数据大到放不下）就是灾难。

**修法**：`np.load(shard_path, mmap_mode="r")` —— 只读需要的 slice，一行改动。

### 2.3 preload 了 10 GB 图像，但 REPA 有 DINO 缓存

`latent_dataset.py:216-225`：`if self.load_image:` → 分配 `(n, 256, 256, 3) uint8`
= 51,036 × 196 KB ≈ **10.0 GB**，并并行解码全部 PNG。

`train.py:890`：`load_image=(args.w_repa > 0)` —— 我们的配置 `w_repa=0.03` → **会预加载**。

但 `REPALoss.forward`（`losses.py:235`）优先走 **DINO 特征缓存**：

```python
feats_cache, missing = self.feature_cache.gather(img_ids, device=...)
if missing:
    feats_cache[_m] = self._teacher_forward(x_0[_m])    # 只有 miss 才用图
```

**→ 缓存全命中时，那 10 GB 图像只被用来提供 `.device`，纯浪费。**

**修法**：
- 校验缓存覆盖率，若 100% 覆盖则允许 `load_image=False`
- 或把 REPA 的 miss 兜底改成"报错"而非"静默用图"，从而彻底去掉 `load_image`
- 或图像改存 **JPEG/缩略图**（日志用不需要 256² RGB 全量）

### 2.4 采样循环里的重复构造

`flow_matching.py:349-381`：
- 每步 `th.full((B,), float(t_i), device=device)` → 可 `ts[i].expand(B)`
- `_tile_kwargs(model_kwargs, B)` **每步重新 cat 条件** → 应移出循环做一次

### 2.5 eval 侧的 O(steps²)

- `in_mem_eval._step_ssim_txt:329` **每个 step 重读整个 summary CSV** → O(steps²)
- `render_poster:504` 每次全量重画所有 step 的图

---

## 3. 冗余（该删）

### 3.1 死文件（`src/eval/` 9 个，约 2,700 行）

引用检查：下列文件**无任何 Python import**，仅出现在别的文件的**注释/docstring**里：

| 文件 | 行数 | 引用情况 |
|---|---|---|
| `auto_eval_gpu.py` | 539 | 仅 `tools/cfg_sweep.py:5` 注释提"已过期" |
| `gpu_batch_eval.py` | 446 | 仅注释 |
| `gpu_batch_eval_v2.py` | 402 | 仅 `tools/cfg_sweep.py:4` 注释提"旧 ControlNet 时代" |
| `auto_eval_ctrl_flow.py` | 334 | 仅注释 |
| `eval_ctrl_ckpt.py` | — | **0 引用** |
| `eval_controlnet_cpu.py` | — | **0 引用** |
| `eval_models.py` | — | 仅注释 |
| `metrics_png.py` | — | 仅自身 docstring |
| `latent_condition_probe.py` | — | 仅注释 |

另有 `auto_eval_cpu.py` / `auto_eval_ctrl.py` 只被 `scripts/legacy_sh/*.sh` 与
`scripts/ops/run_s6_*.sh` 以脚本名调用（无 import）→ **属于 legacy**，建议一并归档。

**建议**：移到 `src/eval/legacy/` 或直接删（git 有历史）。

### 3.2 复制粘贴的指标实现

| 指标 | 份数 | 代表位置 |
|---|---|---|
| **SSIM** | **14** | `inference.py:352`、`eval_auto.py:85`、`auto_eval_ctrl.py:156`、`posters.py:34`、`gpu_ablate_eval.py:47`、`gpu_batch_eval*.py`、`eval_controlnet_cpu.py:47`、`tools/evaluation/make_fame3_poster.py:22` … |
| **VAE 加载** | **11** | `inference.py:59`、`eval_facade.py:43`、`in_mem_eval.py:97`、`auto_eval_cpu.py:144`、`gpu_batch_eval*.py`、`tools/build_std_skel_latents.py:29` … |
| **skel_iou** | **8** | `inference.py:378`、`auto_eval_ctrl.py:224`、`gpu_batch_eval*.py`、`gpu_ablate_eval.py:63` … |
| **LPIPS 加载** | **6** | `inference.py:413`、`in_mem_eval.py:54`、`auto_eval_ctrl*.py` … |
| **MSE** | 4 | `inference.py:348`、`metrics_png.py:32` … |

**修法**：抽 `src/eval/metrics.py`（SSIM/MSE/skel_iou/LPIPS）+ `src/eval/model_io.py`
（VAE/模型构建/ckpt 加载）。**顺带把 §1.1 的 `strict=True` 护栏一次性铺开。**

### 3.3 其它

- `tools/eval_diversity.py:133 build_model` —— 已被 exec cfg_sweep 取代，**旧函数仍留着**
- `dit.py` 2 路 CFG 尾部：
  ```python
  eps = torch.cat([half_eps, half_eps], dim=0)   # 白做 2B 分配
  out = torch.cat([eps, rest], dim=1)
  return out[:original_bs]                        # 又把后一半丢掉
  ```
  → 直接 `torch.cat([half_eps, rest[:original_bs]], dim=1)`
- `dit.py` 双轴 CFG：`yg4 = cat([y_char]*4)` —— `no_char_cond` 下该值被 forward 忽略却仍分配
- `train.py`：`aux_latent_shards_dirs` 字符串在 246/293/900/903/1237/1452 **split 6 次**
- `train.py`：`hasattr(model,'module')` 重复 4 处（553/1606/1686/1716）→ 提 `unwrap()`

---

## 4. 可精简/优雅

### 4.1 `train.py` main() 1630 行

`main()` 位于 **124–1754 行**：分布式初始化 / 目录 / 日志 / 模型构建(261-324) /
预训练+reset+resume(496-877) / optimizer(806-831) / dataset+sampler+loader(882-975) /
scheduler(996-1021) / 早停(1040-1164) / eval cache(1168-1202) / **训练循环(1204-1750)**
全塞在一个函数里。

**建议拆分**：`setup_dist()` / `build_model()` / `build_data()` / `train_loop()` / `ckpt.py`

### 4.2 ★ 150+ 处 `getattr(args, ...)` 且**默认值互相不一致**

同一配置项在不同位置用不同默认值 —— 这是**隐蔽 bug 源**：

| 键 | 位置 A | 位置 B | 位置 C |
|---|---|---|---|
| `vae_downscale` | 201 → 8 | 672 → 8 | **1176 → 4** |
| `eval_cfg` | argparse 2106 → 1.7 | **getattr 1677 → 4.0** | |
| `eval_batch` | argparse 2110 → 16 | **getattr 1674 → 240** | |
| `ckpt_every` | argparse 2089 → 10000 | **getattr 1073 → 5000** | |

→ **归一为 dataclass / pydantic Config**，一次消除全部不一致 + 顺带解决 §1.4 的静默失效。

### 4.3 内嵌类/函数应外提

- `MockVAE`（`train.py:677-691`）
- `_early_stop_check`（90 行，`train.py:1075-1164`）

### 4.4 samplers 抽基类

`DistributedFactorBalancedSampler`(10) 与 `LongEpochDistributedSampler`(67)
各自实现 `num_samples`/`total_size`/`set_epoch`/rank 分片 → 可抽基类。
另外 train.py 还用 torch 的 `DistributedSampler`(953) —— **共 3 条采样路径**，
且 FactorBalanced 用有放回 `multinomial`(48)、**完全不参与 `epoch_steps`/resume 对齐**(921-928)。

---

## 5. 建议动手顺序

| 顺序 | 动作 | 理由 | 成本 |
|---|---|---|---|
| **1** | `heun_batch` 默认改 `False` | 白捡 23.6%+ 的 eval 速度 | 1 行 |
| **2** | `_get_latent` 加 `mmap_mode="r"` | 一行，消除未来的 IO 灾难 | 1 行 |
| **3** | 抽 `src/eval/model_io.py`（`strict=True` 唯一入口），8 处调用点切过去 | **正确性**，且我们已真实踩过一次 | 中 |
| **4** | 删/归档 9 个死 eval 文件 | 减 2,700 行，降低误用 | 低 |
| **5** | 抽 `src/eval/metrics.py`（SSIM 等 14 份 → 1 份） | 后续改口径只改一处 | 中 |
| **6** | `img_id` 改为 CSV 显式列 | 正确性（对 50k 新数据尤其重要） | 低 |
| **7** | `train.py` 拆 main() + Config dataclass | 消除 §4.2 的默认值不一致 | 高 |
| **8** | 采样循环的 `t.expand` / 条件移出循环 | 小幅 | 低 |

**优先做 1/2/6**：三处都是**一行级改动**，收益明确。
**3/4/5 建议合并成一次"eval 子系统整理"**，因为它们互相耦合。
**7 风险最高，建议等 v13 跑起来之后再做**（改动面大，且当前训练正依赖它）。

---

## 6. 已完成（2026-09-17 当天）

| # | 项 | 状态 | 验证 |
|---|---|---|---|
| 1 | **`heun_batch` 默认 `True` → `False`** | ✅ | 微基准 45.16→36.53 ms/步 |
| 2 | **`latent_dataset` 热路径加 `mmap_mode="r"`** | ✅ | 4 处（latent/skel/inst/aux） |
| 3 | **`dit.py` 2 路 CFG 去掉白做的 `cat`** | ✅ | `cat([half_eps, half_eps])`→`cat([half_eps, rest[:B]])` |
| 4 | **新建 `src/eval/model_io.py`** | ✅ | `build_model_from_args` / `apply_post_construction` / `load_model_from_ckpt` / `check_state_dict`；**实测加载 v12 ckpt strict=True 通过**（params 36,550,545） |
| 5 | **`batch_eval.py` 切到 model_io + `strict=True`** | ✅ | 285→265 行；原来是 `strict=False` + **只断言 `unexpected==0`**，missing 全被忽略 |
| 6 | **`cpu_eval_worker.py` 两处切到 model_io + `strict=True`** | ✅ | 含 ControlNet 联训分支 |
| 7 | **`gpu_ablate_eval.py` `build_model` 委托 model_io + `strict=True`** | ✅ | |
| 8 | **`tools/{rerun_eval_wz,eval_sweep_12ch}.py` 改 `strict=True`** | ✅ | 让它们**失败即报错**而不是静默 |
| 9 | **`tools/cfg_sweep.py` 收敛为薄封装** | ✅ | 去掉了重复的构造 + 加载块 |
| 10 | **9 个死 eval 文件移入 `src/eval/legacy/`** | ✅ | 约 2,700 行；附 README 说明理由；2 个 legacy shim 路径已同步 |
| 11 | **`img_id` 统一提取 `extract_img_id()`** | ✅ | 显式列优先 + 正则**锚定结尾** + 失败抛明确错误；实测 HCSU 中文名从"AttributeError 崩"变为"明确报错" |
| 12 | **`inference.make_eval_cache` 用统一提取 + mmap** | ✅ | 失败只警告一次（不刷屏） |

**端到端验证**：`tools/cfg_sweep.py` 在 v12 100k 上跑通 ——
`model loaded strict=True OK` → 采样 → `seen_ssim=0.6945`（20 步）。

### 第二轮（eval 收束 + infra 堵点）

| # | 项 | 状态 | 验证 |
|---|---|---|---|
| 13 | **新建 `src/eval/metrics.py`** —— 指标唯一实现 | ✅ | `mse` / `ssim`（gauss+box 双口径）/ `ssim_torch` / `skel_iou` / `get_lpips` |
| 14 | `inference._mse/_ssim/_skel_iou` 改为**再导出**（向后兼容） | ✅ | **数值逐位一致**（`一致=True`，误差 <1e-12） |
| 15 | `posters._ssim` / `gpu_ablate_eval.ssim_np` 委托到 `metrics` | ✅ | 用 `window="box"` **保住原口径**，box 窗实测逐位一致 |
| 16 | `ssim_torch` 的 padding 从 zero 改为 **reflect** | ✅ | 与 numpy 版差从 **3.2e-3 → 3.8e-5**（纯 float32/64） |
| 17 | **`in_mem_eval` LPIPS 不可用改醒目上报** | ✅ | 原来只 print 一行易漏看；现打 6 行框 + traceback + `lpips_status()` 供 summary 用 |
| 18 | **`_get_callig_map` 路径不存在 → 抛错** | ✅ | 原来静默返回 None → 书家条件静默错位 |
| 19 | **eval shards 覆盖率 <98% → 启动即失败** | ✅ | 原来只打 WARNING 后继续跑 `g=ZERO`（doc68 §2.2 实际踩过） |
| 20 | **采样循环两处重复构造提到循环外** | ✅ | `th.full((B,), t)` → `ts[i].expand(B)`；`_tile_kwargs` 移出循环 |
| 21 | **`aux_dirs_of(args)` 统一 aux 解析** | ✅ | train.py 里 6 处重复表达式 → 1 个函数 |

**第二轮端到端验证**：`seen_ssim=0.6945` —— **与重构前逐位相同**，
确认"14 份 SSIM 收束为 1 份"没有改变任何数值。采样重构后
`heun_batch` True/False 的 `max|diff| = 0.000e+00`（逐位相同）。

### 未做（明确记录）

| 项 | 原因 |
|---|---|
| 剩余 ~80 处 `strict=False` | 多数在 `tools/` / legacy，且**部分是合法的部分加载**（resume / 只灌 backbone）。已提供 `model_io.check_state_dict()` 作为迁移路径，但**不做批量替换**（风险大于收益） |
| `train.py` 拆 main() + Config dataclass（§4.1/§4.2） | 风险最高，且当前 12ch 训练正依赖它 → 等 v13 跑起来再做 |
| `eval_auto.py` 的 torch 批版 `_ssim` 切到 `ssim_torch` | 该文件是 legacy 路径（仅被 `auto_eval_cpu.py` 当库用），等它一起归档时再处理 |
| preload 10GB 图像的浪费（§2.3） | **用户裁定 RAM 充足，preload 可接受** → 不做 |
| `_step_ssim_txt` 每 step 重读整个 summary CSV（O(steps²)） | 影响小（summary 行数 = step 数），留作后续 |

## 7. eval 与 epoch 对齐带来的优化机会（2026-09-17）

### 7.1 现状结构（实测，来自 12ch 运行日志）

| 项 | 实测值 |
|---|---|
| 硬件 | **1 张 RTX 4090，单卡（world_size=1）** |
| **eval 频率** | 每 **5000** 步 —— 恰等于 `epoch_steps`，**正好一个 epoch 一次** |
| ckpt 频率 | 每 **2500** 步（= eval 的 2 倍） |
| eval 耗时 | **39~49 s / 次** |
| eval 跑在哪个 rank | **只在 rank 0**（单卡时即唯一 GPU） |
| eval 输出 PNG | **0 张**（`in_mem_eval_save_samples` 已关）✓ |
| ckpt 大小 | **592 MB / 个** → 40 次 = **23 GB / 100k 步** |
| 跑满 100k 的 eval 总耗时 | 40 × 42 s = **28 分钟**（GPU 全被采样占用，训练停摆） |
| eval 失败处理 | `try/except Exception: logger.warning(...)` → **被吞掉** |

### 7.2 对齐带来的四个具体便利

#### (a) eval 可退化成 epoch 钩子 —— 实现简化 + 失败处理集中
既然**只**发生在 epoch 边界，就不需要在 1600 行的训练循环体里做
`train_steps % gpu_eval_every == 0` 判断。更重要的是**失败处理能集中在一处**，
而不是埋在循环里被 `try/except` 静默吞掉（§1.3 同类问题）。

#### (b) ★ train-only 模式是「等价」的 —— 因为 **eval 点 ⊂ ckpt 点**
- `ckpt_every=2500` **整除** `epoch_steps=5000` → 每个 eval 点都有 ckpt
- 实测核对：eval 步 `{1000..5000, 10000, 15000, 20000, 25000, ...}` **全部**有对应 `.pt` ✓
- `ckpt_keep=None` → 全保留

**→ 事后批量 eval 不丢任何评测点，与在线 eval 完全等价。**

这是"能这么干"的**前提**，而对齐正好保证了它。
反例：若 `ckpt_every=3000` 而 eval 每 5000，则 15000 有 eval 但无 ckpt → 推迟模式直接不可用。

#### (c) ★ 批量 eval 的 GPU 利用率
| | 现在（inline） | 批量（deferred） |
|---|---|---|
| 作业数 | 40 次独立 job | 1 次 |
| 每次样本数 | 60（batch 60） | 2400（batch 240） |
| GPU 占用 | **严重欠载**（37M 小模型 + batch 60） | 接近饱和 |

叠加本轮已做的 `heun_batch=False`（−23.6%）。

#### (d) 训练期不再有 28 分钟的 GPU 空转
那 28 分钟现在给的是「batch 60 的采样」，效率极低；推迟后这段时间训练继续推进。

### 7.3 ★ 关键：这个模式**几乎不用写新代码**

`src/eval/batch_eval.py` **已经就是**"遍历 `results_dir` 下所有 ckpt 批量评测"：

- 逐 ckpt：GPU heun 采样 → VAE decode → 逐样本 MSE / SSIM / LPIPS
- 支持 `--include-steps` 选步、`--force` 重跑
- 输出**幂等追加**的 `eval_stdskel_batch.csv` + `eval_stdskel_summary.csv`（含 P10/Q1/med/Q3/P90）
- 本轮已修好（切 `model_io` + `strict=True`，见 §6）

**→ deferred 模式 = 关掉 `in_mem_eval` + 训练后跑 `batch_eval.py`。**

### 7.4 建议加的两个开关

1. **`--eval-mode {inline,deferred}`**
   - `inline`：现状
   - `deferred`：训练循环只存 ckpt 不 eval；训练后批量跑
   - ⚠ **启动时加一致性断言**：`deferred` 且 `ckpt_every % epoch_steps != 0`
     → 报警"会有 eval 点没有 ckpt，推迟模式不等价"

2. **`batch_eval.py --dit-batch` 默认 50 → 240**（与 `in_mem_eval_batch` 对齐）
   默认 50 是欠载的，直接吃掉 (c) 的收益。

### 7.5 IO 侧剩余堵点（实测）

| 项 | 量 | 评价 |
|---|---|---|
| **ckpt 同步写** | 592 MB × 40 = **23 GB / run** | ⚠ `torch.save` 在训练循环里**同步**执行。单卡时 GPU 停摆等写盘。状态**已经**被 `_state_to_cpu` 搬到 CPU → 可**改后台线程写**，把停顿从"写盘时间"降到"CPU 拷贝时间" |
| ckpt 密度 | 2500（eval 的 2 倍） | 若只关心 eval 点，可降到 5000 → **省一半写盘（23→11.5 GB）和一半停顿**；代价是 loss 曲线回看粒度变粗 |
| eval PNG | 0 张 | ✓ 已关 |
| preload | 一次性 10 GB | 用户裁定可接受 |
| 磁盘 | 1.3 T 空闲 | ✓ |

### 7.6 已实现（2026-09-17）

| # | 项 | 说明 |
|---|---|---|
| 22 | **`--eval-mode {inline,deferred}`** | 两处 eval 块（in-mem + in-process）用 `_EVAL_INLINE` 整段跳过；deferred 时训练循环**只存 ckpt** |
| 23 | **启动一致性护栏** | deferred 下断言 `epoch_steps % min(ckpt_every, epoch_steps) == 0`，不满足 **直接 SystemExit**（附具体修法）。⚠ 判据必须用**实际存盘周期** `min(ck, ep)` 而非裸 `ck`：`ck > ep` 时周期被压到 `ep`，拿 `ck` 判会**误报**（已验证 7 组组合） |
| 24 | **`batch_eval.py --dit-batch` 50 → 240** | 默认 50 在 4090 上欠载，会把批量补跑的全部收益吃掉 |
| 25 | **ckpt 写盘改异步** | `_AsyncCkptWriter`：后台线程序列化 + 落盘，主进程交完 CPU 状态立刻继续训练；`drain()` 在训练结束/退出前 join 并报告失败 |
| 26 | **`_state_to_cpu` 强制拷贝** | 原 `obj.detach().cpu()` 对**已在 CPU 的张量是 no-op**（共享 storage）→ 异步写会读到被训练改坏的数据。改 `.to("cpu", copy=True)`，两种情况都真复制且只复制一次 |
| 27 | **存盘节奏统一到 epoch 边界** | 原三段 `if/elif`（前 5000 步每 1000、之后按 `ckpt_every`，实测落在 1000..5000 + 7500/12500/...，**与 eval 的 5000 边界错开**）→ 统一为 `train_steps % _epoch_steps == 0`。顺带**去掉开头 1000/2000/3000/4000 四次存盘**（每次 592 MB） |
| 28 | ★ **`epoch_steps` 与 `ckpt_every` 合并为一个刻度** | 见 §7.8 |

### 7.9 train.py 拆分：早停独立成模块

`train.py` 2262 → **2274 行**（净变化小，但早停的 130 行闭包从主函数里挪走了）。

| 项 | 说明 |
|---|---|
| 新文件 | **`src/train/early_stop.py`（201 行）** —— `EarlyStopper` 类 |
| train.py 侧 | `_stopper = EarlyStopper(args, checkpoint_dir, logger)` + `_stopper.check()`，**6 行** |
| 收益 | 早停状态机（best/stale/combo/min_delta/去重）从 `main()` 的闭包里独立出来，可单测 |

#### ★ 拆分时发现并修掉一个静默 bug：eval 文件按**字符串**排序

原实现：

```python
ev_files = sorted(glob(os.path.join(checkpoint_dir, "eval_auto_*.json")))
last_ev = ev_files[-1]          # ← 不是最新那一次！
```

文件名是**原始步数**（`eval_auto_95000.json` / `eval_auto_100000.json`），
字符串序下 `'9' > '1'` → **95000 排在 100000 之后**。

实测：

```
字符串序最后一个: eval_auto_95000.json
数值序最新:       eval_auto_100000.json   ← 正确
```

**后果**：跨过 10 万步后早停一直读**旧文件**，`stale` 计数与 `best` 全部失真 ——
而且**不报错**，只是判据永远滞后。（本仓库 early_stop 默认关，所以没暴露过。）

**修法**：`_latest_eval_file()` 按**数值**取最大 step。

#### 过时注释清理

| 清掉 | 原因 |
|---|---|
| 加载顺序注释里的 "inject LoRA" | LoRA 已于 2026-08-31 删除（`src/model/lora.py` 不存在） |
| "full resume works for both LoRA and ..." | 同上 |
| 存盘注释里的 "LoRA 的 delta-only 保存已随 ... 删除" | 同上 |
| `auto_eval_cpu（独立 CPU 进程）` | 该文件已归档到 `src/eval/legacy/` |
| `latent_structure.LatentSkelStructureLoss` ×2 | 路径已从 `src/utils/` 移到 **`src/train/`** |

### 7.8 ★ 合并刻度：`epoch_steps` + `ckpt_every` → 单参数

**动机**：`min(ckpt_every, epoch_steps)` 拼"实际周期"、deferred 还要断言"ckpt 周期整除
eval 周期"、`epoch_steps==0` 的兜底链 —— 这些都是**支持两者取不同值**带来的复杂度。
但两者**语义上本就是同一个东西**（"tick"），实测也从未取过不同值。

**迁移安全性核查**（全仓 121 个配置）：

| | 数量 |
|---|---|
| 用 `ckpt_every` | **121** |
| 两者都配且**相等** | 6 |
| 两者都配且**不等** | **0** |
| 只配 `epoch_steps` | **0** |

→ **保留 `ckpt_every` 作为唯一参数，零配置迁移**；`epoch_steps` 降级为兼容别名
（配了就**必须相等**，不等直接拒绝启动）。

**合并后删掉的复杂度**：

| 删掉 | 原因 |
|---|---|
| `min(ckpt_every, epoch_steps)` 拼实际周期 | 只剩一个数，没有 min |
| deferred 的"ckpt 周期整除 eval 周期"断言（20 行） | eval 点 == 存盘点，**恒成立** |
| `epoch_steps==0` 的兜底链（3 行） | 唯一来源就是 `ckpt_every` |
| 存盘条件三段 `if/elif` | → `train_steps % _epoch_steps == 0` |

**现在的核心就 6 行**：

```python
_epoch_steps = int(getattr(args, 'ckpt_every', 0) or 0)   # 唯一刻度
_es_alias = int(getattr(args, 'epoch_steps', 0) or 0)     # 兼容别名
if _es_alias > 0:
    if _epoch_steps == 0:
        _epoch_steps = _es_alias
    elif _es_alias != _epoch_steps:
        raise SystemExit("...两者已合并为同一刻度，不再支持取不同值...")
if _epoch_steps < 0:
    _epoch_steps = 0
```

**`ckpt_every` 一个参数定义三件事**：
① 每多少步存 ckpt ② epoch 长度（迭代器 reset 点）③ 评测点

**验证**：8 组组合逐一核对（`5000/0→5000`、`5000/3000→拒绝`、`0/5000→5000`、`0/0→0`…）✓

**验证**：
- `_state_to_cpu` 对 CPU 张量确实产生独立副本（`data_ptr` 不同，改原张量不影响副本）✓
- `_AsyncCkptWriter` 连写 3 个文件 → 全部落盘 + `.done` 标记 + 可 `torch.load` 读回 ✓
- 护栏逻辑 7 组组合逐一核对，拒绝的正是真正不安全的组合 ✓
- `--eval-mode` 已注册、`python -m src.eval.batch_eval --help` 可跑 ✓

**预期收益**（12ch 口径）：
- eval 的 **28 分钟 GPU 停摆 → 0**（挪到训练后批量跑）
- 批量后 40 个 batch-60 小作业 → 1 个 batch-240 大作业
- ckpt 写盘停顿从"写盘时间"降到"CPU 拷贝时间"（~2s → ~0.5s）× 40 次
- ckpt 数量减半（2500→5000 密度）+ 去掉 4 次开头存盘 → 写盘量 **23 GB → 约 10 GB**

### 7.7 用法

```bash
# 训练（完全不 eval，只按 epoch 存 ckpt）
python -m src.train.train --config <cfg>.json --eval-mode deferred

# 训练结束后批量补跑（幂等，可断点续跑，可 --include-steps 选步）
python -m src.eval.batch_eval --results-dir assets/results/<exp> \
    --sets seen:assets/eval_seen_v10.csv:10 strict:assets/eval_fame3_strict_clean_v9.csv:50
```

## 附：未做的事

- **未做系统性 profile**（按要求）。§2 的结论来自**代码推理 + 定点微基准**，
  端到端收益需要用真实训练/eval 计时验证。
- **未评估** `src/train/train_controlnet.py`（745 行）与 `src/loss/gaussian_diffusion.py`
  （891 行）是否已废弃 —— 我们用 `flow`，但两者可能仍被 legacy 路径引用。
- **未审计** `tools/` 下约 100 个脚本的存活情况。
