# 风格表征、注入与 Decoder 联训：阶段记录

更新时间：2026-10-26；本报告区分代码、短测和正式训练。**截至本次提交，正式 S/J 长训均未启动，没有新的生成质量结论。**

**最新进度：见第7节。** 第2–6节保留首个提交`ae14347`的阶段快照，未完成项已在续实施中补齐的，以第7节为准。

## 1. 执行范围与变更

最初实现 R0–R4 × I0–I3 的20格及4个跨机复测。后续按新方案缩为 S0–S8 + S2-large，并独立比较 J0–J3；R4预训练暂缓。旧20格测试仅作为运行容量证据，不能冒充新方案已经全验收。

两端统一有效 batch=512，禁止梯度检查点。最新指示：**安全且最快为准，不再以填满20/40GiB作为选型条件，不重复大范围搜索。** 旧结果保留全部候选，直接重选最快安全值；新增 J 分支只做少量针对性短测。

64h指两台并行的自然时间，约128 GPU·小时；尚未锁定最终分配、共同步数和调度 horizon。此前6k→20k→36k属于旧矩阵，不能直接套到新 S/J 方案。

## 2. 各步进度

| 步骤 | 实现内容 | 证据与状态 |
|---|---|---|
| A. 表征与注入 | R0向量、R1多Token、R2空间场、R3几何软路由；I0全局AdaGN、I1空间AdaGN、I2交叉注意力、I3低秩动态卷积 | 真实114M计算图的短测已运行；R4仅保留准备代码，未预训练 |
| B. Writer隔离 | 旧Writer固定null，新路径唯一接收可变Writer；三个中后层注入 | 关闭新路径后无Writer泄漏，匹配原null路径；真实P1短测通过 |
| C. 有效batch累积 | 整批采样时间/噪声/dropout，各份loss按样本数/全batch加权；每步只更新一次Adam、Scheduler、EMA | B256 BF16固定批次检查完成；严格FP32及B512检查未完成，见下文 |
| D. 两端batch搜索 | 四种I分别测试3个候选，每个5步预热+100完整更新 | 两端全部4类完成；旧分配4090的12格复核完成，48复核在范围变更时停止 |
| E. 真实数据短测 | P1@10k EMA严格迁移、真实Top10缓存、STD增强、B512 | 4090完成5步预热+15步测量，1.0328 step/s；不保存科学训练checkpoint |
| F. 新S矩阵 | S0–S8映射已有R/I；S2-large支持512D向量 | 通用代码已写；新规格生成、S2-large容量验收待补 |
| G. J0–J3 | 冻结编码侧，FM与图像分阶段反向，U-Net/Decoder独立裁剪，Decoder raw/EMA随checkpoint保存 | J3真实Decoder+LPIPS反向短测已通过，但46.14GiB高于原安全目标；其余臂及较小图像microbatch待测 |
| H. 长训入口与串行 | 复用现有数据/FM/EMA/checkpoint组件，保存恢复RNG与采样偏移；训练前要求联合loss校准文件 | 入口代码未完成端到端验收；鲁棒串行、校准、配套Decoder评测及64h预算待补，不能开长跑 |
| I. 记录与提交 | 本报告、实现说明和原始JSON入库；state/exp追加最新边界 | 本次提交完成，后续每步另追加记录 |

实现位置：[`src/unet/style`](../../../src/unet/style/)、[`src/unet/joint`](../../../src/unet/joint/)、[`tools/style_matrix`](../../../tools/style_matrix/)。每个源文件均小于1500行。

## 3. 普通FM的B512实测

下表为**此前按显存目标选出的默认值**，最新“最快安全”政策将直接在已有候选中重选，无需重跑搜索。单位为GiB reserved；吞吐为包含反向、Adam和EMA的一次完整有效batch更新。eager/BF16，无重计算。

| 注入 | 4090 micro / 切分 | GiB | step/s | 48 micro / 切分 | GiB | step/s |
|---|---|---:|---:|---|---:|---:|
| I0 | 224 / 224+224+64 | 20.365 | 1.0333 | 448 / 448+64 | 39.111 | 0.9933 |
| I1 | 192 / 192+192+128 | 18.543 | 0.9996 | 448 / 448+64 | 40.816 | 0.9623 |
| I2 | 224 / 224+224+64 | 20.348 | 1.0261 | 448 / 448+64 | 39.100 | 0.9863 |
| I3 | 224 / 224+224+64 | 20.422 | 1.0265 | 448 / 448+64 | 39.314 | 0.9875 |

原始候选：[4090](measurements/family/4090.json)、[48](measurements/family/48.json)；[4090旧12格复核状态](measurements/checks/validation_4090.json)。48的旧状态文件仍写`testing R3I2`，但对应进程已停止，不能按该文件声称仍在跑。现场查询两端均无`python ... tools.style_matrix`进程；4090仅有另一项目约1.1GiB常驻上下文，未操作该进程。

两端软件不同：4090 PyTorch2.5.1+cu121，48为2.14.0+cu126。因此不能把两端吞吐差全归因于模型；后续跨机复测应记录软件栈并比较质量差异。

### 真实P1数据短测

[原始结果](measurements/checks/real_data_smoke.json)。源为4090上的`assets/u100_p1p2_20261025/p1_runs/20261025-113332-Unet-flow/checkpoints/0010000.pt`，严格复制334个张量。Top10 latent覆盖23,485/23,485；R2I0、有效512、micro224。

- 1.0328 optimizer step/s，528.8样本/s；峰值allocated19.832、reserved20.379GiB。
- GPU利用率100%；CUDA event前向347.3ms、反向607.1ms、裁剪/优化8.2ms、EMA2.6ms。
- Writer隔离、非零新增梯度与约0.1的初始注入幅度通过。
- 仅20步短测，不支持收敛、生成质量、长期稳定性或恢复正确性的结论。

### 梯度累积数值边界

[原始结果](measurements/checks/accumulation_b256.json)。相同固定B256，256×1对128×2和64×4，FM loss相对差约0至6e-8；梯度相对L2差约0.48%，Adam更新相对L2差约6.0%。一次optimizer/scheduler/EMA计数符合预期。

历史探针`passed=true`只表示当时较宽BF16阈值通过，**不表示数值等价或严格验证完成**。严格FP32、非等长切分和有效B512仍需一轮小检查；不要重跑庞大测试矩阵。

## 4. S组的可归因设计

| 实验 | 表征 | 注入 |
|---|---|---|
| S0 | R0单向量 | I0全局AdaGN |
| S1 | R1多Token | I0 |
| S2 | R0 | I2交叉注意力 |
| S3 | R1 | I2 |
| S4 | R0 | I1空间AdaGN |
| S5 | R2 Writer×STD CNN场 | I1 |
| S6 | R3几何软路由×Writer基底 | I1 |
| S7 | R0 | I3动态卷积 |
| S8 | R3 | I3 |
| S2-large | R0 512D向量 | I2 |

统一P1@10k EMA、数据/VAE/FM和有效曝光，保留原Script/STD条件，旧Writer路径固定null。S2-large控制每书家参数量；仍保留共享vector→tokens映射，不能宣称其总参数量与S3严格相同。若将512D直接reshape为8×64而不保留映射，会与R1重复，应避免。

## 5. J组：Decoder与图像loss的2×2

| 实验 | Decoder | U-Net监督 | Decoder监督 |
|---|---|---|---|
| J0 | 冻结 | FM | 无 |
| J1 | 冻结 | FM + 生成图像loss | 无；保留对latent的反传 |
| J2 | 可训练 | FM | GT重建loss；不连回U-Net |
| J3 | 可训练 | FM + 生成图像loss | 生成图像 + GT重建loss |

统一P2@20k EMA和原VAE，不与S矩阵交叉。冻结Encoder、quant_conv、post_quant_conv及scale/shift。FM完成后释放其计算图，再对图像microbatch反向，最后一次optimizer更新；两类参数分开裁剪，避免J2的Decoder梯度改变U-Net裁剪比例。

[J3首次容量结果](measurements/checks/J3_first_capacity.json)：B512/FM micro448，图像有效16/micro16，预热3步+测量3步，0.4285 step/s；allocated46.042、reserved46.145GiB。U-Net图像分支梯度变化0.04008，Decoder输出层梯度1.947，冻结编码侧hash未变。使用真实VAE和LPIPS，但U-Net为容量fixture、非科学P2起点，因此这不是训练效果。

该batch**尚未定型**。下一候选是图像micro8×2，保持图像有效16，不开检查点。图像loss权重0.1、重建1.0以及L1/LPIPS/Sobel比例目前只是容量测试值，正式训练需相同起点/样本的梯度标定，不能把默认值写成已校准。

J检查点保存U-Net与Decoder各自raw/EMA及共同优化器状态；验收必须用对应Decoder，防止评测偷偷回到原VAE。该成套评测链尚未完成。

## 6. 下一步与明确停点

1. 在已有B512候选里按吞吐重选安全microbatch，保存选择依据；不重做显存填满搜索。
2. J图像micro8短测及四臂梯度方向/冻结关系检查；检查S2-large容量。
3. 补共同校准、真实GT/cached latent配对、训练保存恢复最小验收和配套Decoder评测。
4. 按测得成本生成两端串行规格、统一训练阶梯与64h预算；串行需锁、身份/hash校验、失败停止、显式阶段推进及保留全部产物。
5. 在“只准备、不长跑”的停点交付。任何长训ETA必须包含评测/保存/加载与余量，不用短测step/s直接承诺64h完成。

公共资产打包代码已写但未执行；48的数据就绪不能视为已完成。一次远程转传因过慢被停止，未解包覆盖原资产；后续使用紧凑、原值校验的数据包。现有checkpoint不删除。

## 7. 续实施：最快安全batch、真实恢复、串行方案

本节记录用户“先提交，再继续实现；batch最快且不爆即可”的后续实施，不覆盖历史结果。

### 7.1 复用候选，未重跑全矩阵

`prepare/select_batches.py`从已有结果按安全上限内最高step/s重选。结果未改变：4090 I0/I2/I3为224、I1为192；48均448。[4090](measurements/family/4090_fastest.json) / [48](measurements/family/48_fastest.json)。后续搜索代码也已移除显存填满条件。

J组只比较J3图像micro8/4，再对J0/J1/J2各做一次5步测量。FM有效512/micro448，图像有效16；每个进程另预热3步，不开检查点。**这是有限候选里的较快选择，不是穷尽搜索得到的理论最优。**

| 实验 | 图像micro | reserved GiB | optimizer step/s | 图像梯度到U-Net | 梯度到Decoder |
|---|---:|---:|---:|---|---|
| J0 | 不使用 | 37.031 | 1.0213 | 无 | 无 |
| J1 | 8 | 39.039 | 0.6541 | 有 | 无 |
| J2 | 8 | 39.637 | 0.6069 | 无 | 有 |
| J3 | 8 | 40.279 | 0.4264 | 有 | 有 |
| J3候选 | 4 | 37.893 | 0.4229 | 有 | 有 |

[原始四臂结果](measurements/joint/selected.json)。micro8比4约快0.8%，短测噪声不支持宣称显著优势；两者均安全，采用8，不再为这点差距加测。图像micro16的旧46.14GiB结果保留，但不采用。

### 7.2 两项累积检查与真实训练保存/恢复

- [FP32](measurements/verification/accumulation_fp32.json)：固定32张，micro32对21+11，TF32关闭。相对梯度误差7.10e-7，Adam更新误差1.21e-4，通过严格阈值。这是算法加权/更新次数的检查，不是吞吐标定。
- [B512 BF16](measurements/verification/accumulation_b512.json)：micro224对128，相对梯度误差0.2257%，Adam更新误差1.8993%，loss误差5.96e-8。不同切分仍非逐位等价。
- [S2-large容量](measurements/verification/S2-large.json)：4090完整114M图、B512/micro224，含前反向、Adam、EMA，5步短测通过；不据此评价生成质量。
- [真实P1恢复](measurements/verification/runner.json)：真实23,485样本缓存与P1@10k EMA，运行1步保存，再恢复到第2步，严格加载raw/EMA/优化器/Scheduler/RNG，优化器和调度均为2次更新。仅2步预检，不是正式S2-large训练。

### 7.3 J3真实GT配对、校准与配套Decoder

4090以FM micro128、图像micro2累计16张做低显存的真实链路验证；不将该预检速度混入48的正式吞吐预算。按`img_id`读取未经额外增强的256图，与缓存latent检查后验mean和重建误差；冻结编码侧hash一致。

[`joint/calibrate.py`](../../../tools/style_matrix/joint/calibrate.py)只做前反向，不优化参数。以原P2@20k EMA计算FM与Image梯度范数，将共同图像权重定为初始FM梯度的10%（上限1），J0–J3复用同一文件；重建权重1单独记录其Decoder梯度。[校准实测](measurements/joint/calibration.json)。这是初始单批次标定，不承诺整个训练期间始终10%。

实测FM梯度0.34434、未加权Image梯度0.82726，因此共同`lambda_img=0.0416239`、`lambda_rec=1`。Decoder侧Image/重建梯度分别3.2913/3.2318，加权后前者约为后者4.24%；这意味着初期Decoder以保重建为主，不能把“U-Net侧10%”误读成Decoder侧同样10%。L1/LPIPS(Alex)/多尺度Sobel比例1/0.1/0.1，全部臂共用。

[真实J3保存恢复](measurements/joint/runner.json)：从P2做1步、保存、恢复第2步；验证U-Net/Decoder优化器状态、Scheduler、raw/EMA和RNG字段。随后导出独立VAE目录，再加载并逐张量核对Decoder EMA，编码侧指纹与scale/shift保持一致。临时外部检查脚本曾因`/tmp`导入路径退出；修正路径后仅补CPU导出验证，没有重跑已经通过的训练步。

[`runtime/evaluate.py`](../../../tools/style_matrix/runtime/evaluate.py)通过checkpoint哈希和raw/EMA标签生成独立VAE目录，校验导出文件哈希，再交给统一评测器。因此VAE路径缓存与评测身份会跟随对应Decoder变化。该导出/加载已经核对；尚未为这两个预检步生成完整测试集图片，不能称为生成质量验收完成。

### 7.4 两端串行与64h预算提案

脚本生成器：[`prepare/specs.py`](../../../tools/style_matrix/prepare/specs.py)；执行器：[`runtime/serial.py`](../../../tools/style_matrix/runtime/serial.py)。生成的两端入口均为`assets/style_matrix_20261026/plan_final/{4090,48}/run_stage.sh`，**默认dry，不启动训练**。

- 4090：S0/S2/S3/S4/S5/S7/S2-large；48：S1/S6/S8及J0–J3。
- S阶梯：**10k → 29k → 43.5k**；J阶梯：**4k → 11.5k → 19k**。从各自最终horizon开始固定LR调度，阶段只改变stop。
- 第二阶段S每端最多2个，保留4种机制时应人工联合比较；J四臂必须都到第二阶段。第三阶段S每端最多1个、J最多2个且保留冻结Decoder对照。不会自动按单一SSIM淘汰模型。
- [预算JSON](measurements/plan/budget.json)显式计入三阶段跨机等待：每阶段取两端较慢耗时再求和，另留**8h**用于准备、保存/加载、评测和余量，预测自然墙钟约**63.95h**。仅按短测推算，不是保证；新权重、GT读取和长训吞吐仍可能改变实际耗时。
- [4090计划](measurements/plan/4090.json) / [48计划](measurements/plan/48.json)。旧`plan_v1/v2`是预算推导产物，不用于执行。

执行器具备：全局GPU锁、外来GPU进程检查、源/数据/代码hash校验、不可变spec/计划、原子checkpoint的`.done`大小+SHA核验、恢复到最新完整checkpoint、阶段评测完成校验、SIGTERM边界保存、失败停止与不自动重试。所有checkpoint保留，每5k及1k/3k和阶段边界保存，不进行自动删除。跨阶段要显式提供晋级文件，原队列不会因新脚本存在而自动恢复。

两端实际`--stage 1` dry入口均已通过，打印`mode=dry`；没有执行`--execute`。[完整生成规格与脚本快照](measurements/plan/specs.tar)保留相同相对路径。训练/评测工具不依赖本工作区尚未提交的旧实验辅助脚本。

### 7.5 仍未执行的工作与当前停点

48所需新资产的紧凑打包/安装，以及两端`ready.json`封存尚未完成。`prepare/ready.py`要求实际资产、严格累积证明、S2-large容量、真实恢复证明及J配对校准/Decoder证明；缺一即报错，不能直接`--execute`。

正式计划尚未启动，未恢复旧V1/三日训练队列，也没有新的模型优劣结论。后续晋级还需要独立Writer/跨字风格评价与实际读图，当前统一评测代码覆盖的是既有生成内容指标，不能凭它自动认定风格改善。
