# 919 / 10 — 数据资产全谱

> 快照日期 2026-09-19。所有数据**只存在于远端** `/root/Workspace/xy/DiT`（不入 git）。

---

## 1. 数据演进史（三代）

| 代 | 名称 | 行数 | 书家 | 字 | 说明 |
|---|---|---|---|---|---|
| 1 | fame3（老数据） | 28,385 | 41 | 4,429 | 首个可用集，strict 平台 0.5680 |
| 2 | **px60**（fame-kxl-tj） | 28,569 | 36 | 4,429 | 剔除坏数据（见 §1.1），v12 系列用它 |
| 3 | **50k_v2**（当前） | **50,786** | **45** | 5,833 | 合并老数据 + HCSU + 新书家，**v13/v14 用它** |

### 1.1 px60 是怎么来的（v12 系列）

起因：训练中部分字（如「多」）生成**中心大黑块**。定位到 `UniCalli` 的
`crop_unicalli.py` 用 column-level 坐标裁剪、与实际图不匹配：

- 4.91% 的 bbox 越界，越界处填黑
- 原图含**黑底白字拓片/反相扫描**
- → 白 canvas 贴黑底 crop = 外圈白 + 中间大黑块

**裁定：直接 drop UniCalli**（修 bbox 成本远高于收益），保留 fame3 + tongji(2,092)。

**唯一判据（用户裁定）**：黑色部分（含外部包络）最粗宽度 > 60px 剔除。
用 `scipy.ndimage.distance_transform_edt(mask<128).max()*2`（256×256）测宽度：

| 宽度组 | 判定 | 依据 |
|---|---|---|
| > 60px | **剔** | 糊成块/残字/超粗 |
| 55~60px | 保留 | 字迹完整只是偏粗（剔会误伤 899 张） |
| 50~55px | 保留 | 完全是正常工整书法 |

**结果**：54,892 → **28,569**（剔 1,045 / 3.53%）。

工具：`tools/build_dataset_px60.py`（过滤+复制）、`tools/build_std_skel_px60.py`（标准字）。

### 1.2 50k_v2 是怎么来的（v13/v14 用）

合并来源：老数据 28,569 + **HCSU 24,855** + bei/tie/calli_tongji 等（见 doc 66）。

**合并后**：53,424 张 / 48 书家 / 5,943 字（字符串口径）。

⚠ **口径警告（doc 73）**：字数有三种口径，必须声明：

| 口径 | 唯一值 |
|---|---|
| `character`（字符串） | **5,821** |
| `character_id` | **7,122**（1,307 个字符对应多个 id） |
| `glyph_id` | **9,232**（2,536 个字符对应多个 glyph_id，**md5 相同故无害**，仅虚高配对统计） |

**最终训练集**：`assets/train_50k_v2.csv` = **50,786 行**（= 51,036 − 250 strict）。
字段：
```
image_path, calligrapher, script, character, calligrapher_id, script_id,
character_id, glyph_id, aug, std_path, source, src_image_path, old_50k_id
```

---

## 2. 评测集（当前口径）

从 50k 里**切**出来的（2026-09-17 修正），切法见 `tools/split_eval_from_50k.py`。

| 集 | 文件 | n | 判据 |
|---|---|---|---|
| **seen** | `assets/eval_v13_seen.csv` | **20** | 训练集内（记忆指标） |
| **strict** | `assets/eval_v13_strict.csv` | **250** | (script,callig,char) 三元组**从没出现过** + 按 script 均衡，**从训练集移除** |

✅ **已核实**：strict 的 250 个三元组在训练集里 **250/250 全为 0**（真未见）。

⚠ **为什么不用旧 eval 集**：它有 5 个书家（39/346/401/483/806）在任何训练数据里都是
0 张 → 50k 的 45 人词表覆盖不了 → 退回原始 id → `y_callig_embedder(806)` 而表只有
46 行 → **CUDA 索引越界**（v13 第一版在 step 5000 就是这么崩的）。

⚠ **strict 的 250 张 latent 仍在 `data/50k/shards_*` 里**（不删），评测端按 img_id
查得到，**不需要重新编码**。

### 2.1 n=50 vs n=250 的采样偏差（重要）

| step | n=50 | n=250 | 差 |
|---|---|---|---|
| 155k | 0.5547 | **0.5703** | **+0.0156** |

→ **任何 n=50 的历史数字都不能直接和 n=250 比**。v12 系列的 strict 全是 n=50，
v13/v14 是 n=250。且偏差**不是常数**（其他 ckpt 实测 +0.004~+0.013，见 `40_results.md` §1.2）。
（注：doc 72 曾把 150k 的 n=50 值 0.5530 与 155k 的 n=250 值 0.5703 相减得 +0.017 ——
同 ckpt 严格对照应为 0.5547 vs 0.5703。）

### 2.2 数据稀疏度（doc 73 实测）

| 指标 | 值 |
|---|---|
| 每书家样本数 | min=99 / med=866 / max=3,131 |
| 每书家覆盖字符数 | min=99 / med=718 / max=1,978 |
| **平均每个书家只覆盖 11.7% 的字** | 88% 的 (书家,字) 组合无监督 |
| **(书家,字) 对恰好 1 张** | **28,235 对（75.0%）** |
| 恰好 2 张 | 6,934（18.4%） |
| 平均张/对 | **1.42** |

⚠ 但**稀疏不是主因**（doc 72 实测）：r(该字的书家数, strict) = **+0.158，只解释
2.5% 方差，且非单调**（0-1 个书家的桶 0.5761 反而比 4-7 个的桶 0.5483 高）。

---

## 3. 关键派生资产

### 3.1 标准字形骨架 g（条件）

**必须是「标准字骨架」，不是「填充字形位图」** —— 首版误建成填充位图
（ink≈0.185 且做了 bbox 归一化），已修。

配方（`tools/gen_base_images.py:render()` 逐行复刻）：

```
字号 200 → 居中 anchor=mm（**不做 bbox 归一化**）
  → skeletonize(a<127)
  → binary_dilation(8邻域, iterations=1)  ≈3px
  → 白底黑线 PNG
```

**质量核对**：ink p50=0.041（p1=0.019 / p99=0.061），与旧 `std_skel3_base_png`
实测 ≈0.033 同域。同 (script,char) 多文件内容 **100% 一致**。
对照图：`data/fame-kxl-tj-px60/_preview/std_skel_cmp.png`

**为什么 3px 不是 1px**（doc 54 实测）：

| | 1px | 3px |
|---|---|---|
| VAE round-trip IoU | 0.861 | **0.960** |
| 字间 raw cos | 0.900 | 0.841（字特异性残差可分） |

SD-VAE 对 1px 细线**不保真** —— 这是把 g 从 1px 切到 3px 的唯一原因。

### 3.2 实例骨架（结构监督 target）

| 资产 | 路径 | 说明 |
|---|---|---|
| 实例骨架 PNG（1px/3px） | `data/fame-kxl-tj-px60/inst_skel{1,3}/` | 由 **GT 图**派生（不是标准字） |
| 实例骨架 latent | `data/aux/inst_skel_latents_px60/` | 447 shards × 64 = 28,569（px60 口径） |

**为什么需要它**（doc 54 实测）：

| 条件 | strict |
|---|---|
| 标准骨架 g | 0.5680 |
| **GT 实例骨架** | **0.7326（+0.16）** |

→ **缺的从来不是"结构"，是"这个书家写的这个字的具体形态"**。

构建：`tools/build_skel_latents.py`（GT→skeletonize→3px 膨胀→VAE encode），224s 跑完。

### 3.3 12ch aux 目标（已证伪，保留资产）

| 资产 | 路径 | px60 覆盖 |
|---|---|---|
| canny latent（白底黑线） | `data/aux/aux_canny_latents_base/` | **100%** |
| skel3 latent | `data/aux/aux_skel3_latents_v8/` | 92.7%（缺 tongji 2,084） |
| skel3 latent（px60 全量） | `data/aux/inst_skel_latents_px60/` | 100% |

⚠ **极性铁律**：canny 与 skel 必须**统一白底黑线**。历史上 canny 存黑底白线
（`cv2.Canny` 原生），靠 wz latent 生成时 `invert=True` 补救 → 任何绕过 wz 直接读
PNG 重编码的脚本会重踩这个坑（doc 58 已修：`edges = 255 - cv2.Canny(...)`）。

### 3.4 书家条件表（四代）

| 代 | 文件 | 形状 | pairwise cos | 有效秩 | 评价 |
|---|---|---|---|---|---|
| 老 | `assets/callig_emb_pretrained_base.pt` | (52,128) | — | — | fame 41 家 |
| v13 | `assets/callig_emb_pretrained_50k.pt` | (45,128) | 0.0204 | 34.4/45（76%） | 对比学习 |
| v14 | `assets/callig_script_emb_pretrained.pt` | (87,128) | 0.0135 | 77.2/87（89%） | 层级 SupCon |
| **v15** | `assets/multistyle_k4_pretrained.pt` | **(87, 4×384)** | pair 内质心 cos 0.884 | 18.0/87 | **DINO K-Means**（每 pair 4 个模态质心） |

**⚠ 重大数据事故（2026-09-19 发现，已修）**：DINO CLS 提取器（`tools/extract_dino_cls_50k.py`
及 base 版）的 `load()` 里 `(1,1,H,W).repeat(3,1,1)` **参数数 < 张量维数 → 每张图都抛
RuntimeError**，被 `except: return zeros` **静默吞掉** → 所有图变成同一张全零图 →
**全库 50,786 行 CLS 特征完全恒定**（unique=1）；另有缺 `/255` 的第二层 bug（attention 饱和）。
后果：

1. **v13/v14 SupCon 表的 "DINO 质心锚定" 从未生效**（所有质心相同，锚定项为常数）
   —— 上面两代表里的健康几何（cos 0.0135 / 有效秩 89%）**完全是标签层级驱动的**，
   与 DINO 内容无关。v14 作为"87 vs 45 标签粒度"A/B 仍成立，但"表与 DINO 对齐"的说法作废。
2. REPA 的 DINO cache **不受影响**（`build_dino_cache.py` 用的是 `/255*2-1`，正确）。
3. 修复：两个 extractor 重写 `load()`（interpolate→/255→repeat，失败**打印并丢行**，
   batch 循环 feat/标签对齐，保存前唯一值自检）；`build_multistyle_k4.py` 加**垃圾特征守卫**
   （采样 2000 行查唯一值，<100 直接拒跑）。
4. 特征已重提（失败 0/50786、采样唯一值 1998/2000），v15 表已重建：
   pair 内 4 质心平均 cos **0.884**（同 pair 模态差异真实但不大）、有效秩 18/87
   （DINO 特征被书体/总体结构主导）、簇均衡 med 0.169、3 个 <4 样本的 pair 平铺。

**v15 表怎么建的**（`tools/build_multistyle_k4.py`）：

```
1. 重提特征: tools/extract_dino_cls_50k.py → assets/dino_cls_50k.npz (50786, 384)
2. 逐 pair K-Means(K=4): L2 归一化特征, kmeans++ + 4 重启, n<K/空 pair 平铺/书家级回退
3. 产物: embedding (87, 4*384) 直接灌 MultiStyleEmbedder.embedding_table
         centroids (87,4,384) / pair_mean (87,384)  <- --style-anchor-mode mean 的锚定目标
```

**v13/v14 表的问题（历史）**：45/87 行、每行 1 个向量、而且冻结 → 苏轼 3131 张多模态风格压成
1 个固定向量 → strict 0.3485（r=−0.62，见 `00_overview.md` §3.2）。

### 3.5 其他辅助资产

| 资产 | 路径 | 说明 |
|---|---|---|
| REPA DINO 缓存（50k） | `data/dino_cache/50k_v1/` | 51,036 patches，9.3 GiB fp16 |
| REPA DINO 缓存（px60） | `data/dino_cache/base_sym_v1/` | 161,766 patches，29.6 GiB |
| latent shards（图） | `data/50k/shards_img/` | VAE f8，4ch×32×32 |
| latent shards（g） | `data/50k/shards_std/` | 标准字骨架 |
| latent shards（eval g） | `data/fame-kxl-tj-px60/shards_std_eval/` | **必须独立**（eval 的 img_id 与 train 不同套） |

---

## 4. 数据侧的坑（全部踩过）

| # | 坑 | 后果 | 修法 |
|---|---|---|---|
| 1 | **g / aux shards 用 train csv 的 img_id 建，eval 是另一套** | strict 命中 0/237 → g 全零 → decode(0) 出灰黄棕、指标失真 | 独立 `eval_skel_latent_shards_dir` |
| 2 | **极性不一致**（canny 黑底 vs skel 白底） | 结构通道语义相反 | 统一白底黑线（doc 58） |
| 3 | **uid 双重叠加**（`aug_base_sym.py`） | 4,892 个 uid 冲突 / 8,584 行错图 | 已修 |
| 4 | **标准字建成填充位图** | g 的 ink 0.185 vs 0.041 | 按 render() 逐行复刻 |
| 5 | **UniCalli bbox 越界** | 中心大黑块 | 直接 drop 该来源 |
| 6 | **dino cache 按旧 id 复用** | REPA teacher **静默错位**（新 id 命中旧 id 的特征） | 每代数据重建 cache |
| 7 | **`glyph_id` 一字多 id** | 虚高 (书家,字) 对数（39,267 vs 35,879） | 无害（md5 相同），统计要声明口径 |
| 8 | **`character_id` 一 id 多字**（10 例） | 若当字符身份用会串字 | no_char_cond=true 下无影响 |
| 9 | **DINO extractor 的 `repeat` 维数 bug 被 `except: return zeros` 吞掉** | 每张图变全零图 → **全库特征恒定**，SupCon 锚定全程失效 | 重写 load()（见 §3.4）；**兜底 except 必须打日志/计数，否则静默全灭** |

---

## 5. 数据检查工具

| 工具 | 用途 |
|---|---|
| `tools/check_g_coverage_px60.py` | g 覆盖率（防 strict 假崩） |
| `tools/verify_std_px60.py` | 标准字骨架形态核对 |
| `tools/check_std_identity.py` | 同 (script,char) 多文件一致性 |
| `tools/split_eval_from_50k.py` | 切 strict/seen（按 script 均衡 + 三元组未见） |
| `tools/stat_by_source.py` | 按来源统计 |
| `tools/stat_skel_ink.py` | 骨架 ink 分布 |
| `tools/scan_polarity.py` | 极性筛查 |
| `tools/check_two_skel.py` | 两套骨架对照 |