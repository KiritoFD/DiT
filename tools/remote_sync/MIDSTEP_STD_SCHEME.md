# V3C 方案：标准字库作为「去噪中线目标」（MIDSTEP_STD）

> 目的：解决用户观测到的「推理时字可能给错」——数据链接层面已审计无误（见 §5），
> 根因更可能是模型尚不足以在去噪过程中稳定跟随标准字形条件。因此**不改采样端**，
> 只在训练损失里，于中间噪声水平额外加一项「去噪结果逼近标准字形 latent」的监督，
> 把字形结构在去噪中段锚定。

状态：**已实现并在远程 v3c 运行中**（step≈580 起步下降）。

---

## 1. 一句话方案

在扩散训练中，当样本落在设定的**中间噪声带**（`sqrt(α̅) ∈ [alo, ahi]`）时，
额外监督：
```
loss_std_mid = MSE( model 预测的 clean latent x0_pred , 标准字形 latent g )
```
主损失仍从 GT x0 学报内容 + 书家风格；`loss_std_mid` 只在中段噪声步把
x0_pred 拉向同字的标准字形 latent，使字形结构在去噪中途被钉住。采样端不变
（纯噪声初始 + CFG 只 drop callig + 标准字形 `g` 作为 token-add 恒在）。

## 2. 为什么是「中线」而不是「初始点」

- 初始点混合（HYBRID，`glyph_init_mix`）在去噪最噪声的一拍注入标准字形，
  结构先验强但可能锁死、损伤书家风格，且只在采样端生效、训练不管。
- 中线目标（本方案）在**训练时**就教会模型：当我处于加噪一半的状态，我要
  让预测结果朝标准字形 latent 靠拢。它把字形结构与书家风格解耦——内容由标准
  字形锚定，风格由主损失从 GT 学。采样端无需改动。

## 3. 实现（train.py）

新增 CLI：
| 参数 | 默认 | 含义 |
|------|------|------|
| `--w-std-mid` | 0.0 | 中线目标权重（0=关；建议 0.3~0.8，偏小防抹风格） |
| `--std-mid-alo` | 0.35 | 中间噪声带下界（`sqrt_alpha_cumprod`） |
| `--std-mid-ahi` | 0.75 | 中间噪声带上界（`sqrt_alpha_cumprod`） |

训练循环内，在计算完 `pred_xstart` 后：
```python
loss_std_mid = zeros
if w_std_mid>0 and pred_xstart is not None and model_kwargs.get('g') is not None:
    sqrt_a = tensor(diffusion.sqrt_alphas_cumprod, device)
    mid = (sqrt_a[t] >= alo) & (sqrt_a[t] <= ahi)      # (N,)
    if mid.any():
        loss_std_mid = ((pred_xstart[mid] - g[mid])**2).mean()
loss += w_std_mid * loss_std_mid
```
- 仅当 `use_glyph_cond` 开启且批次带 `g` 时生效（与 v3b 的条件来源一致）。
- 归一化到该中段子集做平均（不稀释到全 batch）。
- `pred_xstart` 与 `g` 都在同一 `×0.18215` 缩放域（与 GT latent 同域），可直接比。
- 记录 `StdMid` 到日志 `(step=...) StdMid: raw ...`。

## 4. 配置（v3c）

`exp_v3c_XL_glyphcond_midstep_kailishu.json`：
- `w_std_mid: 0.8`, `std_mid_alo: 0.35`, `std_mid_ahi: 0.75`
- 其余与 v3b 相同（X L/highdim 2cond, glyph_scale_init=0.4, batch 16, 30k, lr 1e-4）。
- `results_dir: 5script/results/v3c_xl_glyphcond_midstep`

启动（远程）：
```bash
tmux new-session -d -s exp "bash _launch_exp.sh exp_v3c_XL_glyphcond_midstep_kailishu.json exp_v3c_midstep.log"
```

## 5. 对齐审计（数据链接）

已对 `kailishu_train.csv`(51098) 与 `kailishu_eval.csv`(200) 全量审计，**0 错**：
- `glyph_id == script_id*7026 + character_id`：全部正确。
- `std_glyph_key` 的字库（kai=楷 / li=隶）与 `script_id` 对应：全部正确。
- 每个 `std_glyph_key` 对应的 `std_glyph_latent/{kai,li}/U+XXXXX.npy` 文件：全部存在。
- show5 固定样本（鼎/商/也/刻/昌）的 `ord(char):05X` 键与本地/远程库一致。

结论：条件 `y_char`(glyph_id) + 标准字形 `g` + 主 loss 的 GT `x0` **三者对同一字**，
无「字给错」的链接层错位。剩余差异（若推理仍错字）来自模型跟随能力，正是本方案要改进的。

## 6. 观测

step→ 实测（远程 log `exp_v3c_midstep.log`）：
| step | Diff | StdMid | Total | X0Lat |
|------|------|--------|-------|-------|
| 300  | 0.769 | 2.432 | 2.715 | 900 |
| 380  | 0.612 | 1.903 | 2.134 | 434 |
| 500  | 0.322 | 0.677 | 0.863 | 100 |
| 580  | 0.312 | 0.311 | 0.561 | 43 |

StdMid 快速下降说明模型学会了在中段让 x0_pred 逼近标准字形；Diff/Total 同步下降，
训练正常收敛。

## 7. 后续可调项

- `w_std_mid` 过高会过度锚定字形、损书家风格（当前 0.8 偏激进，必要时降到 0.3~0.5）。
- 中间噪声带 `[0.35,0.75]` 可调：带越窄越聚焦单一中线信噪比。
- 收敛后可取样验证 show5 字是否更稳，再决定是否沿用、或叠加 HYBRID 初始点。
