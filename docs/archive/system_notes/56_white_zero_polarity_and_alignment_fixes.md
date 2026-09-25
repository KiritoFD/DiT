# 56. 白底归零 / 通道极性 / ref 对齐 —— 之前的实现错在哪、以后怎么办

> **触发**：2026-09-14 用户裁定「正确启用白底归零、对齐 ref、hidden 512、RoPE 关闭」。
> 本文是**错误复盘**：逐条写清「之前错在哪 → 怎么发现的 → 代价是什么 → 怎么修的」，
> 末尾给出**防复发守则**（这是本文最重要的部分）、新 run 配置与待验证项。
>
> 相关：51（aux 结构通道逐行核对）、52（ref 主模型精读）、55（base 数据线与 Sp/2 启动）。

---

## 0. 一句话结论

之前的实现有 **3 个真实缺陷**（canny 极性反、白底未归零、eval 构建器漏传新参数）和 **3 处与 ref 的口径偏离**（aux 权重、通道顺序、RoPE）。
缺陷的共同性质是：**都是"沉默的"——不报错、训练照跑、loss 照降，但目标空间被污染**。
其中「eval 构建器漏传」导致 sty16 训练 21,900 步**评测全程零产出**，是最贵的一次。

---

## 1. 错误 E1：canny 通道极性反了（黑底 vs 白底）

### 1.1 现象

三个扩散目标通道的底色/极性**不一致**。实测 per-channel latent 均值（`rebuild_latents_wz.py` 打印）：

| 通道 | per-ch mean | 底色 |
|---|---|---|
| `image` | `[1.02, 0.65, 0.15, −0.63]` | 白底 |
| `skel3` | `[1.33, 1.16, −0.01, −0.90]` | 白底 |
| `canny` | `[−0.88, −1.66, 0.64, 0.76]` | **黑底（符号全反）** |

### 1.2 根因（代码级，纯疏漏）

`tools/gen_base_images.py` 里两条派生分支的极性处理**不对称**：

```56:61:tools/gen_base_images.py
        sk3 = binary_dilation(sk, ST, iterations=1)
        Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(f"{SKEL_OUT}/{iid}.png")
        import cv2
        edges = cv2.Canny(a, 80, 180)
        Image.fromarray(edges, "L").save(f"{CANNY_OUT}/{iid}.png")
```

- skel 分支：显式 `np.where(sk3, 0, 255)` → **白底黑线** ✓
- canny 分支：直接 `Image.fromarray(cv2.Canny(...))` → `cv2.Canny` 输出本身是 **边缘=255 / 背景=0**，即 **黑底白线** ✗

同一函数里相邻两行，一条做了极性归一、另一条直接落盘原始输出。**没有报错，图看起来也"像边缘图"**，所以一路没被发现。

### 1.3 代价

12ch 等权 MSE 下，canny 通道的主要梯度来自「把整体亮度翻过来」这个**与内容无关的常数变换**。模型被强迫浪费容量去补偿一个本可在数据侧一行代码解决的问题；同时三个通道 bias 方向不一致，进一步恶化了下一节 E2。

### 1.4 修法

`tools/rebuild_latents_wz.py` 的 `JOBS` 表加了 `invert` 标志，对 canny 先 `255 − x` 再编码：

```49:54:tools/rebuild_latents_wz.py
JOBS = [
    ("img",   None,                          "data/latents/final_latents_base_wz",  False),
    ("skel3", "data/skel/final_skel3_base",  "data/skel/aux_skel3_latents_base_wz", False),
    ("canny", "data/aux/final_canny_base",   "data/aux/aux_canny_latents_base_wz",  True),
]
```

**注意**：只重建了 latent，**没有改 PNG**（`data/aux/final_canny_base` 仍是黑底）。
即"canny PNG 黑底"这个事实仍然成立——**任何绕过 wz latent、直接从 PNG 编码的推理脚本都会踩同一个坑**。

---

## 2. 错误 E2：白底未归零（latent 里藏着一个大常数 bias）

### 2.1 现象

三通道 latent 的空白区域数值 ≈ 白底常数（见上表，量级 1.0 上下），**不是 0**。而真正的笔画/边缘信号是叠加在这个大 bias 上的小偏差。

### 2.2 根因

ref 的 `train_moyun2.py` 里有 `custom_zero` 机制，但**ref 的启动脚本全部为 0，从未启用**；我们照抄配置时沿用了"不减"。
ref 不减、也不加回（`generate.py` 直接 decode），两边自洽；**我们照着"不减"抄，同时又引入了 canny 黑底（E1），问题被放大成"三通道 bias 方向还不一致"。**

### 2.3 代价

- 扩散模型相当一部分容量用于「重建白底常数」，而非笔画结构；
- t→0 时真正的残差被大 bias 淹没，**有效动态范围被压缩**（这是出发点，收益需实验验证，见 §5）。

### 2.4 修法（含一个必须成对的联动）

训练侧：三类 latent 全部减白底 latent（`data/white_latent.npy`，确定性 `latent_dist.mode()`，与训练侧其他 latent 构建方式一致）。

推理侧：**decode 前必须加回**，由 config 开关 `aux_zero_white` 控制：

```313:322:src/eval/in_mem_eval.py
                # 白底归零 (aux_zero_white): 训练目标已减去白底 latent,
                # decode 前必须**加回**, 否则整幅图偏色。
                if bool(getattr(args, "aux_zero_white", False)):
                    _wl = _WHITE_LAT_CACHE.get("w")
                    if _wl is None:
                        _p = "data/white_latent.npy"
                        if os.path.exists(_p):
                            _wl = torch.from_numpy(np.load(_p)).float().to(_lat.device)
                            _WHITE_LAT_CACHE["w"] = _wl
                    if _wl is not None:
                        _lat = _lat + _wl[None]
```

> **这是本轮最容易踩的坑**：ref 的推理脚本**不加回**是正确的（它没减）；
> 我们一旦启用减白底，**不加回就整图偏色**。
> 「减什么就必须加回什么」是训练的**不变量**，不能用"照抄 ref"绕过。

---

## 3. 错误 E3：加新模块时漏改 eval 构建路径（最贵的一次）

### 3.1 现象

容量扫描 `c41x_sty16` 训练跑到 **21,900 步**（Diff 0.3101，健康），但 **eval 产物为空**——
`checkpoints/eval_auto_*.json` 一个都没有，worker 在 16:26→16:29 之间持续 `rc=1` 重试。

### 3.2 根因

worker 报 `AssertionError: main weights unexpected=5` —— 正是新模块引入的 5 个 key
（`style_proj.*`、`style_role`）。**`src/eval/cpu_eval_worker.py` 的模型构建器没有传 `style_token_n`**，
构建出来的模型不含这些子模块，加载 ckpt 时它们变成 unexpected，`strict` 断言直接崩。

```163:172:src/eval/cpu_eval_worker.py
        callig_style_attn=bool(a.get("callig_style_attn", False)),
        callig_n_style=int(a.get("callig_n_style", 8)),
        # 风格 token 每层注入 (GlyphStyleCrossAttn): 漏传会使 ckpt 的
        # style_proj.*/style_role 变成 unexpected -> assert 崩 (2026-09-10 修)
        style_token_n=int(a.get("style_token_n", 0)),
        style_role_init=float(a.get("style_role_init", 0.02)),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch)
```

### 3.3 代价

**21,900 步的评测全部丢失，三个容量实验的结论不可用**（这是"容量扫描失败"的真正原因——
不是训练坏了，是**评测从来没跑起来**）。而且失败是"静默重试"形态：日志里只有循环的 `rc=1`，
不主动去看 worker 日志就发现不了。

### 3.4 修法

补传参数（已上传）。**但更重要的是流程**：新增模型参数时，必须全仓扫描所有构建点。

```bash
grep -rn "DiT_2Cond_models\[" src/
```

---

## 4. 偏离 E4：与 ref 的三处口径不一致（本轮对齐）

| 项 | ref | 我们（之前） | 本轮 |
|---|---|---|---|
| aux loss 权重 | **等权**（`mean_flat` 对 12ch 整体） | `0.3,0.8`（自创） | **`1.0,1.0`** ✓ |
| 通道顺序 | `cat(image, edge, skeleton)` | `cat(image, skel, canny)` | **`image, canny, skel`** ✓ |
| RoPE | `if_rope=False`（默认关） | 开 | **`rope=0`** ✓ |
| `learn_sigma` | True（输出 24ch） | False | 保留差异，**注明口径** |
| 扩散类型 | DDPM（eps） | flow（velocity） | 保留差异，**注明口径** |

**注意第 1 行是有代价的**：54 号文档曾实测"12ch 等权时结构通道吃 ~51% final-layer 梯度"，
据此**自创**了 0.3/0.8 权重。但严格核对 ref 后确认：ref 就是等权，**0.3/0.8 是我们的发明，不是 ref 的做法**。
→ 现在的选择是「先严格对齐 ref 拿一个干净基线」，权重扫描留作后续单变量。

---

## 5. 更深一层的错误 E5：用"参数幅度"推断模块重要性

48 号里我看到 `learned glyph_scale = 0.1896`（init 0.6，模型主动调小 3 倍），
推断"输入层 token-add 无关紧要、可以删"。

**GPU 推理时消融实测直接推翻**：

| 配置 | ssim | skel_iou |
|---|---:|---:|
| baseline | 0.7838 | 0.2227 |
| **`no_glyph_add`** | 0.5021 | **0.0108** |
| `no_xattn` | 0.5412 | 0.0217 |
| `no_g` | 0.4331 | 0.0116 |

去掉它骨架遵循度几乎归零。**zero-init 模块"学到的小幅度"≠不重要**——
`glyph_scale` 是固定幅度直通、`xattn` 的 `out_proj` 是 zero-init，两者互补（粗注入 vs 精细寻址）。

**教训**：判断模块可删必须做**推理时消融**，不能看参数量、梯度幅度或学习到的 scale。

**教训续（E6）**：同一批消融还发现 **CFG 一直用错了**（历史全部用 0.7）：

| cfg | ssim | skel_iou |
|---:|---:|---:|
| 0.7（历史全部） | 0.7838 | 0.2227 |
| **2.0** | **0.8261** | **0.3517** |

skel_iou **+58%**。评测超参本身是实验变量，**换模型/换目标通道数后必须重扫**。

---

## 6. 防复发守则（本文核心）

1. **多通道数据必须做极性/底色体检**：派生图（skel/canny/edge）生成后立刻打印 per-channel mean，
   **符号不一致就是极性错**。不要相信"图看起来对"。
2. **常数偏移必须"减-加"成对出现**：训练减什么，推理就必须加回什么，
   且必须写成显式 config 开关（如 `aux_zero_white`），不能是隐式假设。
3. **改模型必须同步所有构建点**：`grep -rn "DiT_2Cond_models\[" src/`，
   train / eval worker / daemon / sweep / 推理脚本一个都不能漏。新参数导致的
   `unexpected=N` 会**静默重试**，不会主动报警 —— 新 run 起来后必须**主动确认 eval 产物真的在生成**。
4. **对齐 ref 要逐项列表核对**：通道顺序、权重、归一化、位置编码、输出头、扩散类型，一项一行，
   明确标"已对齐/保留差异+原因"，禁止"大概一致"。
5. **判可删必须推理时消融**：不看参数量/梯度幅度/学到的 scale（E5）。
6. **评测超参是实验变量**：cfg / 采样步数 / 样本量任一变动，都必须重扫并**在文档写清口径**
   （E6；n=10 的 ±0.01 波动量级内不要下结论）。
7. **配置项要与生效路径一致**：`use_char_cond=false` 时 `num_characters`/`char_embed_dim`/
   `char_proj_mode`/`freeze_char_table` 全部不生效，留在 config 里纯属误导（见 48 号 §4）。

---

## 7. 本轮最终配置（已启动）

`src/train/configs/v11_pretrain_Sp2_base_wz.json`：

| 项 | 值 |
|---|---|
| 模型 | `DiT-2Cond-Sp/2`（h512 / d12 / heads8，~59M） |
| 数据 | `assets/train_base_noaug.csv`（54,892；含 tongji），**从头** |
| 目标 | 12ch = `cat(image, canny, skel3)`，**顺序对齐 ref** |
| aux 权重 | **1.0, 1.0**（等权，对齐 ref） |
| 白底归零 | **启用**（三类 latent 减白底；`aux_zero_white=true` 推理加回） |
| canny 极性 | **白底**（`255−x` 后编码） |
| RoPE | **关**（`rope=0`，ref 默认） |
| 注入 | 输入层 token-add + `ZeroAdaLNInjection` ×4 |
| 扩散 | flow velocity（**与 ref 的 DDPM 口径不同**） |
| LR | cosine，warmup 3000，total 400k，batch 192 |
| eval | `in_mem_eval` 每 2,500 步（seen n=10 / strict n=50），带白底加回 |

**产物**：
- latent：`data/latents/final_latents_base_wz` / `data/skel/aux_skel3_latents_base_wz` / `data/aux/aux_canny_latents_base_wz`
- 白底：`data/white_latent.npy`
- 日志：`logs/v11_pretrain_Sp2_base_wz/train.log`

启动确认（step 600）：

```
Diff: 0.6505 | c12[img=0.7869 canny=0.6267 skel=0.5380] | REPA(w=0.03): 0.0124
4.45 steps/s | Mem 19.70G
```

---

## 8. 待验证（诚实标注）

1. **白底归零的收益本身尚未验证**。ref 从未启用它；我们启用是采用**你的裁定**，
   理由（动态范围/信噪比）在理论上成立，但**没有对照实验**。
   若后续要归因，需要一个「wz vs 非 wz」的同配方对照。
2. **通道顺序对齐 ref 对指标有无影响**：理论上无（模型对通道排列无先验），但未验证。
3. **canny PNG 仍是黑底**（只重建了 latent）：任何直接读 PNG 的脚本仍会引入 E1。
   若要彻底消除，应修 `gen_base_images.py` 并整体重生成。
4. **`learn_sigma` / DDPM-vs-flow 两处差异仍在**，因此本 run 与 ref 的指标**不可直接比**。
5. cfg 口径：本 run 的 in-mem eval 用 cfg=0.7（历史口径），但 48 号实测最优在 ~2.0 ——
   本 run 的第一个 eval 点出来后，建议补一次 cfg 扫描再定主口径。
