# 2026-10-27：延长轮实现与双机重新拉起

## 决策与范围

00:53延长轮决策（S8、J3、成对长训、J组5k→12k）按"停止现有训练、整体重新串行拉起"执行。旧计划`cold50k/4090`与`deep85_decoder/48_v2`停止并保留存档；新计划根目录`assets/style_matrix_20261027/matrix64/{4090,48}`。

## 代码改动（已同步两机）

- `src/unet/joint/objective.py`：新增内部臂`J4`——图像监督+可训练Decoder、无GT重建分支。外部J3→内部J4；J3−J1检验解冻、J2−J3检验重建锚。J4的Decoder梯度仅来自生成latent的图像分支。
- `src/train/step.py`：新增`build_restart_scheduler`——受控两阶段LR restart，锚定绝对步数，peak=0.5×原峰值、warmup 200、cosine覆盖延长段（S 25k / J 7k）。
- `tools/style_matrix/runtime/train.py`：支持spec级`continues_identity`（延长spec跨身份恢复5k/50k checkpoint）与`schedule_restart`；identity.json允许新旧两种身份。
- `tools/style_matrix/runtime/serial.py`：评测协议按`arms`过滤（J用raw空间187字、S用top10-row val100）；大资产（>1MB）封存只记path+size，不哈希（用户决策：64h运行不对GB级文件重复哈希）。
- `tools/style_matrix/prepare/matrix64.py`：新计划生成器。复用旧spec字节级副本（S0/J0/J1/J2）保证checkpoint可续；生成S8=(repr2,inj3)、J3（内部J4）、S_ext75/J_ext12延长spec；迁移旧run目录；封存ready；生成run_stage.sh。
- `tools/style_matrix/probes/i2_token_check.py`：S2/S3的KV token预检。

## 计划结构

4090 stage1：S0(续1k)→S1→S4→S6→S8，各50k；stage2：S1/S4/S6/S8候选延长至75k（selection.json人工选定一队，run_stage.sh轮询24h）。48 stage1：J0(续5k)→J1(续575)→J2(续50)→J3(新)各5k→S2/S3/S5/S7各50k→J0–J3延长至12k。wall_hours均64h作硬上限，超时从队尾（延长段）按序截断，与保护顺序一致。

## 探测实测（compiled，200步/100步窗口）

| 机器 | 臂 | step/s | 峰值GiB | 备注 |
|---|---|---|---|---|
| 4090 | S0 | 1.614（前次实测） | 19.2 | 正式队列从1k续训 |
| 4090 | S1 | 1.550 | 20.8 | |
| 4090 | S4 | 1.542 | 21.0 | |
| 4090 | S6 | 1.577 | 20.0 | DynamicConv无速度损失 |
| 4090 | S8 | 1.566 | 20.4 | R2×I3组合原生支持，fm曲线正常 |
| 48 | J0 | 1.752（正式实测） | 18.3 | 已完成5k |
| 48 | J1 | 0.880 | 38.6 | 编译后比旧eager估算0.65快 |
| 48 | J2 | ~0.51（前次50步实测） | 39.8 | |
| 48 | J3 | 0.765 | 39.8 | 首次运行即验证：recon分支0样本、decoder梯度0.022>0 |
| 48 | S2/S3/S5/S7 | 探测中 | — | |

S8/S6与S1/S4同速，4090队列名义估算56.6h（64h上限内）。S2 KV预检：8个KV token明显可区分（token_std≈0.98、token01距离≈1.34），attention熵2.032<均匀2.079——初始化接近均匀属预期，架构无"单token退化"问题。

## 资产与协议（登记）

- p2_train.csv两机SHA一致（976b573d…）；`generic48_kxl_top10_20261022/std_bbox64/index.csv`两机SHA一致（c02cef37…，15,000键）——S组协议同源。
- 48此前缺`u100_pretrain_20261024/`下val100.csv与char/font remap（KB级补齐）及std_shape bank。bank按[Data.md](../../Data.md)配方在48**本地重建**（build_rasters逐位下采样校验+precompute_bank，seed90，GPU约50秒）：**重建结果与4090的bank SHA256完全一致（bc3db05f…）**——零GB级传输实现比特级同协议。
- 配对校验两次按设计拦截错误组合：bank_64（为exp-std store构建）配bbox64 store → "font cache and STD store index disagree"。这是`StdShapeDataset`的预期保护，不可绕过。
- 字体：`tools/fonts/`下5个TTF（~50MB，无重建路径的源资产）按例外同步到48；index.csv中的绝对字体路径用`build_rasters --path-remap /root/...=/home/ds/...`机器本地重映射，store索引文件本身未改动。训练与bank预计算不打开字体，仅build_rasters渲染需要。
- 48磁盘131G可用：48的S组checkpoint节奏为5k/10k/25k/50k四处（ckpt_every=25000），预计总写入约100GB内。

## 4090 S0停滞

停机前抓取主进程+8 worker堆栈：全部`futex_wait_queue`用户态等待、内核栈为空（[存档](stall_stacks_4090_20261027.txt)）；无py-spy/gdb，Python层根因未定位。重新拉起后S0从1k续训已越过1350旧停滞点，判断为暂态死锁；若再现按数据触发处理（改换seed重跑）。

## 执行事故记录

- serial封存校验按设计拦截了三次"封存后改代码"：matrix64估算修复（4090）、serial.py大资产口径更新（48）均触发"Sealed source changed"；用最终代码重封后通过。教训：代码冻结点应在封存前明确。
- 被杀任务遗留的serial在训练但其内存封存与新.py不匹配（会在下次call()才失败）——发现后立即杀掉重拉，损失约1分钟。
- pkill模式含自身命令行导致ssh会话自杀一次；改用`style_matr[i]x`模式。
- 48曾按"机内自有bank_64"配置S组，探测被配对校验正确拒绝；改为本地重建u100 bank（SHA与4090一致）后通过。2.8GB跨机中转方案已中止，登记为反面案例，规则固化进[AGENTS.md](../../AGENTS.md)。

## 最终状态（03:20）

- 4090：matrix64队列运行中，S0从1k续训已越过旧停滞点（5,450+，约1.55–1.61 step/s），5k评测已触发。
- 48：matrix64队列运行中，J0@5k完成并通过heldout187评测，J1从575续训（0.880 step/s）；估算全程53.0h，全自动。
- 探测临时checkpoint已清理（两端probes仅保留日志与train.jsonl）。

## 遗留与后续

- 48 stage1全自动；4090 stage2需在50k评测后写入`matrix64/4090/selection.json`（格式`{"2":{"4090":["S1","S4"]}}`，候选S1/S4/S6/S8选2，run_stage.sh轮询24h）。
- 延长段统一受控restart：peak2=0.5×峰值、warmup200、cosine到0.1×peak2（S 25k / J 7k）；首步保持原LR保证连续性。
- 机制验收与推理消融、187字终评协议、交叉Decoder解码按1027.md延长轮专节执行。
