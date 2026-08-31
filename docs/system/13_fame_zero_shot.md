# fame 管线：数据模型、模型资产与零样本结论（2026-08-30）

## 1. Data Model

### 1.1 fame 数据集（构建：tools/build_fame_fast.py）

| 项 | 值 |
|---|---|
| 来源 | archive/final_manifest.json（329,715 张全量）按书家白名单选取 |
| 名单 | 44 规范书家（57 名单含合并：赵孟→赵孟頫、郑燮→郑板桥、孫過庭→孙过庭） |
| 规模 | train 51,322 + eval 500（严格凸包切分） |
| 极性 | 全部白底黑字（翻转 2,751+9 张，备份 flip_backup*/） |
| 脚本 | 楷/行/隶/草/篆/六体（草篆六体无标准字库覆盖） |
| 字体资产 | /tmp: simkai(楷) STXINGKA(行) SIMLI(隶) |

### 1.2 派生产物

| 文件 | 内容 |
|---|---|
| `fame.npz` | 全量 img VAE latents（52,518×4×32×32 f16 + img_ids） |
| `final_latents_fame/` | 20 shards（训练加载用，按 img_id 索引） |
| `final_skel3_fame/` | 全部 3px 骨架 PNG（51,822） |
| `final_skel_latents_fame/` | skel latents（20 shards，**GPU 编码**） |
| `final_skel_latents_fame_std/` | **标准字库骨架** latents（训练条件，20 shards） |
| `skel_bank_train.npz` | 推理库：训练集每字一张骨架 latent（14,372 字） |
| `skel_bank_std.npz` | 推理库：标准字库骨架 latent（eval 覆盖 238/483） |
| `5script/train_fame.csv` / `eval_fame_strict.csv` | 训练/评测表（严格凸包：两要素覆盖+组合未现+图未训） |

### 1.3 评测协议（重要）

- 训练/评测的 skel 条件域必须匹配（GT 书法骨架 ≠ 标准字库骨架）
- 统一 cfg=0.7（骨架条件下 CFG>1 有害，单调劣化）
- 报告用中位数；骨架跟随度用「输出笔画 vs 条件骨架」的膨胀 IoU（ dilation=4 ）

## 2. Models

| 模型 | 训练 | 数据 | 条件 | 结果（各自口径） |
|---|---|---|---|---|
| s21 fame 预训练 | 从零，150k 上限 | fame 51,322 | callig+char ID | early-stop @40k，best ssim 0.4647（fame_strict） |
| **fame-ctrl**（GT 骨架） | warm-start s21@30000，50k | fame + GT skel latents | GT 书法骨架 | **SSIM 中位 0.8045 / MSE 0.104 / 失败 2.4%**（n=500）；骨架跟随 IoU 0.798（GT 条件）、**0.82-0.87（标准字库条件）** |
| fame-ctrl-stdskel | warm-start s21@30000，30k | fame + 标准字库 skel latents | 标准字库骨架 | **未训通**：跟随 IoU 0.09-0.19，条件被忽略（见 §4） |

## 3. 零样本评测结论（fame-ctrl @50000，n=100，cfg0.7，本地 4070）

| 骨架条件 | SSIM vs 该 GT | 跟随 IoU（vs 给定条件） | 解读 |
|---|---|---|---|
| eval 图自己的 GT 骨架 | 0.8202 | 0.798 | 高保真复刻 |
| 训练集同字骨架（他人书写） | 0.5074 | （未测，推断高） | 正确跟随了给定骨架 → 与本 GT 结构不同 → SSIM 低是应得的 |
| 标准字库骨架 | 0.5201 | **0.82-0.87** | **zero-shot 可用**：字形正确、风格由书家条件决定 |
| 无骨架 | 0.5007 | — | base 水平 |

SSIM-vs-GT 在换骨架时下降是**正确行为**（跟随了给定的骨架），不是失败。
零样本推理流程：**字 → 标准字体渲染 → 骨架化 → latent → fame-ctrl → 书法字**，已验证 8/8 字形正确（豈密遂等）。

## 4. fame-ctrl-stdskel 为什么没训通（重要教训）

标准字库骨架对模型而言是**冗余条件**：char ID（DINO 嵌入）已隐含字形，
ControlNet 注入分支在「控制信号可由其他条件推出」时会学会**门控掉注入**
（梯度上忽略条件是局部最优）。GT 骨架训练能成，是因为书法骨架携带
char ID 之外的实例/风格信息。佐证：该模型用 GT 骨架评测 ΔSSIM≈-0.005
（条件域互换后互不响应）。

可能的解法（未实施）：训练时大幅提高 char 条件 dropout（迫使模型依赖
骨架）、对骨架条件加噪声增广、或改用 char ID 不可用的训练设定。

## 5. 本次修复的 bug 清单

1. `train_controlnet.py` ckpt 只存 `ctrl_encoder.*` → `injections.*` 丢失
   （已修：`not startswith("main.")`；resume 同步修）
2. eval daemon 相对路径 → 永不消费 pending（已修：绝对路径启动）
3. fame 数据 2,751+9 张反色图（已翻转， latent/骨架级联重编码）
4. fame eval 骨架 stale（/tmp 路径， 已补 9/9）
5. eval csv 组合泄漏（已改严格凸包切分）
6. npz 重复解压爆内存、heredoc 引号、pkill 自匹配等工具链问题

## 6. 文件索引

- 本地评测：`_diag/local_zero_shot.py`（4 臂）、`_diag/skel_follow_test.py`（跟随度）
- 远程量化：`/tmp/std_ctrl_quant.py`、`/tmp/decisive_follow{,_gt}.py`
- 数据：`fame_meta.json`、`fame_flip2.json`、`skel_banks_meta.json`
- 指标：`5script/fame_final_eval.json`、`5script/fame_zero_shot_eval.json`、
  `5script/fame_ctrl_watchdog.json`
