# P2条件利用审计：4090执行记录

## 顺序与身份

修统一评测 → 固定P2 EMA → 正确/错误/null Writer FM审计 → 固定噪声CFG扫描 → 独立评价及实际读图 → 通过后才蒸馏。
不重训架构、不继续普通长预训练；本轮只用4090。当前未启动蒸馏。

固定checkpoint：`assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt`。
产物根目录（4090）：`/root/Workspace/xy/DiT/assets/p2_audit_20261025/`。
`identity.json`保存checkpoint SHA256、CSV SHA256、EMA、STD sigma、求解器身份。

## 原生基线

统一评测、val100、EMA、Heun50、STD sigma=0：

| SSIM | LPIPS | Ink-IoU | Frag | SoftInk |
| --- | --- | --- | --- | --- |
| 0.592304 | 0.349359 | 0.280905 | 2.303146 | 1.044938 |

来源：`baseline/val100/eval_samples_ctrl/step0020000/eval200fix/metrics.json`，不同于此前sigma=.65在线曲线。
独立PNG评分读取量化图像，原生基线在浮点图像上计算；没有宣称逐位重放。CFG只在同一PNG评分器内部比较，不混用两行数字。

## 错误与根因

1. 离线汇总把集合嵌套字典当标量字典过滤，导致空结果；DONE复用也可能返回空字典。
2. 旧Writer-swap自行按0→1积分，网络时间没乘1000，把灰度PNG切成窄条；生成前还加载无用LPIPS。
3. `build_probe.py`的VAE输出已为0–1，送DINO时再次除255。旧probe训练/评测输入域不一致，本轮GT sanity仅18%，旧风格数值作废。
4. 远端`artifacts.py`旧版本缺`write_png_index`，无法写生成清单；已同步现有实现。
5. 名为`style_bank_top10.npz`的文件实际仅有0–5六个组号，旧质心循环按十书家取均值会产生NaN；拒绝把它当十书家评价器。新评分只使用已核验类别的probe。

## 修复

- 离线汇总解包集合、恢复已完成metrics，核验样本量及关键有限值；Top10 raw/row映射幂等，拒绝混合标签空间。
- 新`generate.py`只调用主线`sample_latents`及FlowMatching，不独立实现ODE；固定STD/Script，仅替换Writer。
- Writer-swap限定该书体实际训练过的Writer组合；同字符同噪声；PNG保持256×256×3。
- 生成结束后才独立加载DINO/LPIPS；完整记录CFG分支计算成本，不把双分支说成免费推理。
- 修复probe的0–1契约，重建到`writer_probe_fixed.pt`，旧文件保留；评分器拒绝未经修正的probe。

## 测试与防复发

四个小型unittest已在4090通过（1.4秒）：标签映射幂等、嵌套/DONE指标恢复及样本量校验、常速度FM oracle验证1→0及t×1000/CFG分支、饱和黑墨图的双向墨量匹配。
命令：`python -m unittest tools.p2_audit.test_contracts`，不跑冗余全库测试。
新probe带`input_contract`及class顺序；评分每次报告GT sanity；文件存在不等于风格评价可信。

## 设计及验收

100图、98字符；追溯P0/P1/P2后，本批没有完全未见字符，68图为未见Writer×Script×Character组合。
确定性posterior mode、同图固定epsilon；t=.05至.95分10桶比较正确Writer、同书体合法错误Writer平均、null Writer。
delta=错误/null损失减正确损失，正数才支持正确Writer更好；按字符分组bootstrap。

CFG=1/1.2/1.4/1.6同图同噪声，测SSIM/LPIPS/Ink-IoU/Frag均值及中位数、SoftInk、EDT ridge笔宽W1及Writer margin。
首轮结束后再决定分时间CFG、多seed及墨量匹配控制；不自动把最高probe分当作教师。

验收需风格margin跨seed稳定改善，内容/LPIPS/笔宽不明显退化，并通过实际读图及独立内容/墨量控制。
DINO logistic是图像随机划分，并非严格跨字/跨来源教师；GT sanity正常也不足以单独批准蒸馏。
若只增粗、只提升自身分数或收益不确定，停在审计阶段，保留P2，不启动蒸馏。

## 完成结果与决策

原生基线、FM十桶、CFG四档、基线/CFG1.6三个seed、两段时间窗、120张Writer-swap、软墨量匹配及全部100图读图均已完成。
没有修改P2权重。队列状态`completed_no_distillation`；**本轮未通过可靠风格教师门槛，不启动蒸馏**。

三seed平均（同批100图的三个噪声实现，不是300个独立字符）：

| 指标 | CFG1 | CFG1.6 |
| --- | --- | --- |
| SSIM（PNG） | .591128 | .592880 |
| LPIPS | .349015 | .346177 |
| Ink-IoU | .280953 | .289914 |
| Frag均值 | 2.556083 | 2.050997 |
| 笔宽W1 | .026018 | .025172 |
| SoftInk | 1.078530 | 1.097322 |
| 修正probe Micro Top-1 | 28.33% | 39.67% |

三个seed的probe margin原始增量为+.02682/+.02881/+.03512，字符分组CI均为正。
分数式灰度腐蚀/膨胀将每图软墨量匹配回对应CFG1后，margin增量为
+.01869（CI +.00098～+.03737）、+.02011（+.00291～+.03854）、+.01764（−.00314～+.03889）。
每图最大墨量残差小于.0005，没有悄悄排除匹配失败的图。
这支持收益不全来自墨量，但第三seed不确定；控制本身改变局部几何，不是纯粹的因果风格分离。

修复probe的随机图像留出准确率47.5%；本批原始GT仅24%，VAE重建GT32%，不能充当可靠书风真值。
不能将probe的+11.3个百分点直接写成真实书风提升+11.3个百分点。

时间窗仅默认seed：高噪声t∈[.5,1]用CFG1.6，其他时间CFG1，LPIPS .345833 / IoU .292116 / probe36%；
全程为.345972/.292446/35%。未证明窗口更优，但采样阶段17.17秒 vs全程22.44秒；基线11.67秒。
含尾批padding的等效分支NFE分别169.12/224/112，不能称为免费计算。

FM在t=.85/.95，正确Writer优于错误Writer平均，delta约+.0643/+.0863；
但null在.05～.85桶的平均MSE低于正确Writer，直到.95反转。Writer信息存在，不等于各时段都改善MSE。
不同t的delta不能直接当同强度的图像收益，一步clean误差还乘t²。

实际读图：全部100图中，CFG多为加黑、增粗及局部边缘变化；缺笔/错误结体往往不修复，部分偏粗字继续变重。
Writer-swap同字同噪声产生明显结构和笔宽变化，条件并非不起作用；但一些Writer使字极细、缺笔或偏离目标写法。
“响应强”不等于“方向正确”。120张swap的probe Top-1约30.8%，受上述评价器限制。

保留CFG1.2～1.6和高噪声窗口为推理候选，**不替换默认checkpoint、不宣布最优、不启动训练**。
下一关是可信的独立书风/内容验收，不是继续加大CFG或更改结构。

本地图及记录：

- [全部对照第1页](../assets/p2_audit_20261025/comparison_page0.png)
- [第2页](../assets/p2_audit_20261025/comparison_page1.png)、[第3页](../assets/p2_audit_20261025/comparison_page2.png)、[第4页](../assets/p2_audit_20261025/comparison_page3.png)、[第5页](../assets/p2_audit_20261025/comparison_page4.png)
- [Writer-swap](../assets/p2_audit_20261025/cfg/swap/writer_swap.png)
- [原始指标及身份记录压缩包](../assets/p2_audit_20261025/audit_evidence.tar.gz)
