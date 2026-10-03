# DiT_2Cond 模型与 4-way 条件 dropout

> 对应源码：`src/model/dit.py`（DiT_2Cond 约 line 400-1055）、`src/model/__init__.py`
> 训练入口参数：`src/train/train.py` 的 CLI（`main_from_cli`）与配置文件

## 1. 模型总览

- **DiT-2Cond-S/2**：patch size 2，latent (4,32,32) → 256 tokens，hidden 384，12 层，~33M 参数。
- 输入：`x`(latent) + `t`(时间步) + 两个条件 id：`y_callig`（书家）、`y_char`（字符/glyph）。
- `learn_sigma=True`：输出 `2×in_channels`（8 通道）；flow 取前 4 通道作 velocity。
- 条件融合两种模式：
  - `factorized_add`：`y_emb = (callig_proj(e_callig) + char_proj(e_char)) / √2`，低维可组合，**默认**。
  - `xl_highdim`：拼接后过 `cond_fusion` MLP（XL 对齐，保留预训练 adaLN 结构）。
- 字符表可选 **DINO 初始化 + 冻结**（`char_dino_embeddings` + `char_dino_index` + `freeze_char_table=true`）：`glyph_id = script_id*7026 + character_id`，用每个「字×书体」的样本 CLS 均值初始化嵌入行。

## 2. 4-way 条件 dropout（CFG 的四种训练信号）

CFG 需要「部分条件缺失」的样本；本系统比标准 2 路（有/无）多出一路单因子缺失，形成 **4-way mask**（`factorized_add` 与 `xl_highdim` 分支各有相同实现）：

```python
r = torch.rand(N)
drop_all = r < cond_drop_all_prob
drop_one = (cond_drop_all_prob <= r) & (r < cond_drop_all_prob + cond_drop_one_prob)
which_glyph = torch.rand(N) < cond_drop_which_glyph_prob
y_callig = where(drop_all | (drop_one & which_glyph),  num_classes, y_callig)  # drop callig
y_char   = where(drop_all | (drop_one & ~which_glyph), num_classes, y_char)    # drop char
```

| 分支 | 条件组合 | 学到的分数 | 概率 |
|---|---|---|---|
| **full** | callig + char | joint score | `1 - p_all - p_one` |
| **drop_all**（uncond） | 无 | CFG 基准 | `p_all` |
| **glyph-only** | 只有 char | content score `s_G`（字符内容分） | `p_one × p_which` |
| **callig-only** | 只有 callig | style score `s_A`（书家风格分） | `p_one × (1 - p_which)` |

新增可配置参数 **`cond_drop_which_glyph_prob`**（bool 型丢弃偏向）：

```python
parser.add_argument("--cond-drop-which-glyph-prob", type=float, default=0.5,
    help="drop-one 时选择 drop callig (→glyph-only) 的概率; 书家维度样本充足, "
         "字符维度才是难点, 建议 >0.5. 0.5=均匀.")
```

### 2.1 为什么默认偏置 0.75（s19 配置）

数据统计（mid-clean，见 `05_dataset.md`）：

- **书家仅 67 类**，每类样本 138~11022（中位 1152）→ 类别少、样本足，容易学。
- **字符 5461 类**，每类样本 6~112（中位 18）→ **稀疏、才是难点**。

若 `p_which=0.5`（均匀），drop-one 预算（25%）对半劈给两类，等于「67 个书家和 5461 个字符拿同样的专门训练预算」—— 书家维度被过度过量供给。s19 配置：

| 参数 | 值 | 效果 |
|---|---|---|
| `cond_drop_all_prob` | 0.05 | uncond 5% |
| `cond_drop_one_prob` | 0.25 | drop-one 25% |
| `cond_drop_which_glyph_prob` | **0.75** | → **glyph-only 18.75%**（5461 字符内容分）/ **callig-only 6.25%**（67 书家，且 70% full 样本本就含书家信息） |
| 合计 | | full 70% / glyph-only 18.75% / callig-only 6.25% / uncond 5% |

参数全部经 `resolved_config.json` 持久化到实验目录（见 `06_training.md` §5），可随时用 json 调整比例重启。

### 2.2 校验

- `cond_drop_all_prob + cond_drop_one_prob <= 1`（构造时校验）。
- dropout 仅在 `self.training` 时生效；推理/eval 走 `forward_with_cfg`。

## 3. CFG 推理（forward_with_cfg）

```python
def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, g=None):
    # 批次翻倍: [cond | uncond]，uncond 用 embedder.num_classes 行（embedding 表有专用 uncond 行）
    # 对输出只取 [:in_channels] 前缀做 CFG 组合（velocity/eps 子空间），sigma 通道原样保留:
    half = uncond + cfg_scale * (cond - uncond)
    return cat([half_over_eps, rest])
```

- uncond 行 = `y_embedder` 表的 `num_classes` 索引（不参与训练形状），embedder 的 forward 带 `train` 标志避免把 dropout 应用到 uncond。
- **flow 下不 clip**；cfg 作用于 velocity 无界场。
- 推理默认 `eval_cfg=1.7`（flow 最优区间；s19 配置固定于 `resolved_config`）。

## 4. 工厂与加载

- `DiT_2Cond_models["DiT-2Cond-S/2"]` 等工厂字典；`src/model/__init__.py` 统一再导出，旧 `from models import DiT_2Cond_models` 仍可用。
- `load_main_model(...)`（在 `src/model/controlnet.py`）：以构建参数重建主模型 + 加载 ckpt（优先取 `ema`，缺省回退 `delta`/裸 state_dict），`strict=False` 并打印 missing/unexpected —— 这是 ControlNet warm-start 的装配入口。

## 5. 与 3cond 的差异

`DiT_3Cond`（callig+script+char，老版本）保留兼容：参数 `y_script`、`num_scripts`、`cond_drop_one` 三分支 `which∈{0,1,2}`。当前所有新实验走 2cond。