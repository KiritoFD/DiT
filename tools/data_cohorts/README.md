# 固定书家梯度数据与训练

2026-10-21 冻结版本：`assets/data_cohorts/top20_20261021_v1/datasets_final`。
不使用后续 `scan_thinframe` / `release`：细线边框不再扩大清洗范围。

| 书家数 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 2 | 6226 | 20 | 40 |
| 5 | 14069 | 50 | 100 |
| 10 | 23485 | 100 | 200 |
| 20 | 30386 | 200 | 400 |

训练与测试按书家前缀嵌套；所有级别训练排除全局验证、测试和旧评测保留组。
评测每张图固定噪声种子，使相同样本跨子集的 GT / STD / 初始噪声一致。
明显质量问题、重复泄漏组和标签歧义已隔离；不宣称经过完整人工真迹鉴定。

## 入口

- `prepare.py`：汇总候选、扩展原图和统一字符标识。
- `build.py`：分组留出、嵌套划分与条件表。
- `ready.py`：核验缓存/真实加载器接口，生成不带预算的配置模板。
- `train_serial.py`：冻结可执行配置与输入哈希，顺序训练、失败即停。

实际运行地点：4090 `/root/Workspace/xy/DiT`。
队列目录：`assets/data_cohorts/top20_20261021_v1/launch_70k`。
配置/身份：`queue.json`、`top20.json`、`top10.json`、`top5.json`、`top2.json`。
动态状态：`status.json`；各组启动日志：`topN.log`。
`started.lock` 防止重复启动；失败后必须人工确认，不自动重跑或跳过失败组。

## 本轮配置

**21:41修订（当前生效）**：旧70k队列已停止。新队列 `launch_30k_lr7e5`，
max_steps / lr_decay_steps=50000，stop_at_step=30000，LR7e-5。
四组都从零先训练30k（15.36M曝光）后切换；其余设置不变。
输出改为 `assets/results/clean_cohorts_20261021/launch_30k_lr7e5/topN_u114`。
下列70k/LR6e-5描述保留为原始计划，不再生效。

Top20 → Top10 → Top5 → Top2，各自从零训练 70000 步，B512，即每组 35.84M 样本曝光。
普通 U-Net `[192,384,576]`、每尺度 Encoder/Decoder 各4个 ResBlock，约114M参数；
保留共享 Writer/Script AdaGN，不启用硬分离、Writer-only 分支、额外 Gate 或 DiC。
FM、官方 SD VAE、LR6e-5、warmup1500、cosine到70k、EMA0.999；REPA/Critic关闭。
STD形状强扰动关闭，原软STD模糊保留。每5k保存并评测，验证集选点，测试集不参与调参。
训练目标缓存为 posterior mode、scale0.18215、shift0；同一全局缓存可包含验证目标，
但训练样本严格只由各自 train.csv 决定，不从缓存自动扩展训练集。

114M 是已有同曝光指标更强的候选，不代表视觉风格必然优于72M。
REPA未做此 U-Net 的同配置消融；关闭是复用已验证基线，不是证明无效。
固定曝光不等于各组相同 epoch 数；这轮比较的是相同训练计算预算下的数据规模效应。

## 2026-10-22 测试矩阵

`python -m tools.data_cohorts.evaluate --output assets/audits/cohort_matrix_30k_20261022_v2`

固定30k EMA / CFG1 / Heun50 / 原始测试CSV / 每图噪声；
Top20测20/10/5/2，Top10测10/5/2，Top5测5/2，Top2测2，共10组。
小batch4、VAE batch1，不编译、不修改在训模型；保存所有PNG、逐图指标和协议哈希。
Top2的30k权重未就绪时等待训练队列完成该组，不拿中途权重代替。
输出`results.csv`、`results.json`、`status.json`和每组`identity.json`、原生评测目录。
结果只在同一测试列上跨模型比较，不比较不同测试规模的总均值来排名。

## Bug 分析报告

### 一、错误现象
首次独立测试调用抛出`unexpected keyword argument diffusion_override`；复现：旧版远端
`run_in_mem_eval(..., diffusion_override=...)`。输入检查通过，无PR归因；训练不受影响。

### 二、根因分析
自顶向下追踪发现本地评测接口支持额外参数，但远端稳定接口不支持。
影响仅新建测试脚本的首次调用；不涉及权重、划分或训练循环。

### 三、最小化Diff格式修复补丁
```diff
- run_in_mem_eval(model, ev, 30000, 'cuda', results_dir=str(out), diffusion_override=...)
+ run_in_mem_eval(model, ev, 30000, 'cuda', results_dir=str(out))
```
采样器仍通过公共eval_args的flow/heun/50配置创建，不覆盖训练源码。

### 四、防回归测试用例
```python
import inspect
from src.eval.in_mem_eval import run_in_mem_eval
inspect.signature(run_in_mem_eval).bind(None, None, 30000, 'cuda', results_dir='x')
```
实际部署前执行签名绑定，随后用完整400张测试集做集成验证；缺失LPIPS或样本数不符直接报错。

### 五、预防建议
独立工具使用两端共同公共接口；部署时在实际运行环境核验，而非假设本地与远端源码完全一致。
