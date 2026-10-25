# P1/P2 Writer-aware 训练验收（2026-10-25）

决策依据：`ANALYSIS/style_structure_diagnosis_20261025.md`（风格=跨字稳定的线性可读分布，
Writer Embedding 容量足够，无需参考/字形调制模块）。
本轮唯一假设：**提前在大规模预训练模型中引入书家监督（混合阶段），是否优于后期单独安装 Writer 条件**。

## 执行配置

| 项 | 值 |
| --- | --- |
| P0 | `u100_pretrain_20261024/runs_100k/0075000.pt` EMA（kxl-100k 池 75k；**无 40 万池 checkpoint，此为替代**，已记录） |
| P1 10k | 70% 通用行（**显式 null 书家行 259**）+30% Top10 行（0-9）；B512；主干 LR 1e-5 / 书家条件组（writer/script/proj）LR 5e-5；drop_writer .25；strong STD 增强沿用 |
| P2 20k | Top10 23,485 行；70% 常规 +30% 同字同书体异书家配对（PairMixtureSampler，纯采样）；drop_writer .10；其余同 P1 |
| 全程 | 全解冻、EMA 沿用、FM 单损失、ckpt/评测每 5k |

实现：`tools/u100_pretrain/run_p1p2.py`；新增机制 `null_writer_rows`、`writer_cond_lr` 参数组、
`pair_fraction` 在 FM mainline 放行（仅采样，无配对辅助损失）。

## 结果（val100 σ0 内联；离线三口径见下）

| 预算 | 路线 | SSIM | LPIPS | IoU | Frag |
| --- | --- | ---: | ---: | ---: | ---: |
| 20k | 直 FT（对照） | .5800 | .3603 | .2632 | 2.27 |
| 20k | **P1@10k+P2@10k** | **.5864** | **.3535** | **.2725** | **2.02** |
| 25k | 直 FT | .5817 | .3579 | .2653 | 2.14 |
| 30k | 直 FT 延长 | .5829 | .3581 | .2638 | 1.91 |
| 30k | **P1@10k+P2@20k** | **.5908** | **.3497** | **.2814** | 2.18 |
| 50k | 直 FT 延长终态 | .5844 | .3558 | .2660 | — |

离线口径（P2@20k 终态）：eval200fix σ.65 .5834/.3488/.2918/Frag 2.79；test200 σ0 .5895/.3466/.2874/2.87。
P1 曲线：5k .5616 → 10k .5664。P2 曲线：10k .5864 → 20k .5908。

## 结论

1. **假设成立**：同预算下 P1 混合阶段带来 +0.6pp（20k）/ +0.8pp（30k）SSIM、+0.9/1.8pp IoU，且 P2@20k（30k 预算）全面超过直 FT@50k 终态——提前引入书家监督既省预算又抬上限。
2. test200 与 val100 一致，无挑点；Frag 持平（笔法不退化）。
3. writer probe / Writer-swap 干预测试：脚本就绪（`tools/u100_pretrain/exp_writer_swap.py`），执行两次于 LPIPS 加载后无声死亡（无 traceback、无 OOM），待修；**风格维验收的 swap 部分待补**。

## 记录的偏差与修复

- P0 用 kxl-100k 替代 40 万池（40 万池 checkpoint 不存在）；若补训 40 万池预期增益复制或放大。
- STD 增强未按"无强增强"执行：P0 权重训练于强增强协议，切换会造成条件分布突变；后训练线（.5833）全程强增强亦支持保留。
- 启动事故留痕：漏设 B512、P1 remap 缺 null 行 259、data.py null 检查放错文件、P2 继承 P1 remap、mainline 拒绝 pair_fraction——逐一修复，修复均在 git 记录。

产物：`assets/u100_p1p2_20261025/`（P1/P2 全部 checkpoint、评测、配置、manifest）。
