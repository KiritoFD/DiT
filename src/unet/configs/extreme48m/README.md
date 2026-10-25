# Extreme Deep-Narrow U-Net：下一轮配置

2026-10-22。独立书家表已训练并审计完成。17:22正式启动X1、B512：
训练输出 `assets/results/extreme48m_top20_x1/20261022-172221-Unet-flow`。
compiled容量短测13.61GiB allocated、14.94GiB reserved、2.69step/s；正式长训吞吐仍待首批记录。X2尚未启动。
实际配置保存于 `assets/results/extreme48m_top20_x1_launch_20261022_compiled/train_config.json`。
此前eager B512预检OOM，B256短跑随后停止并保留日志；这暴露了预检必须与正式编译模式一致，不能混报容量。

## 架构和固定契约

通道96/192/256，Encoder和Decoder每尺度8个ResBlock，Middle两个：共50个ResBlock、5个Attention。
实际构建48,325,988参数，其中48,325,316可训练；冻结的21×32 Writer embedding包含20个真实书家及CFG-null行。
保留原FM、VAE、STD-soft、Writer/Script融合，不加入REPA、Critic、Writer-only或DiC。

独立表使用 `assets/data_cohorts/top20_20261021_v1/writer_tables_supcon_v1/callig_table.npy`：
20×32单位范数原表，不使用适配旧30k生成器的旋转/范数匹配版本。
主训练及EMA必须保持表不变，恢复checkpoint后再安装外部表，校验ID映射并清除旧Adam槽位。
推理没有GT条件，仍为STD、Writer ID、Script ID、噪声。

数据仍是冻结的 `datasets_final/top20`：train30386、validation200、test400。
STD形变增强保持off，与上一轮实际基线一致；附件的strong不与架构消融同时引入。

## 两组配置

| 项目 | X1 | X2 |
| --- | --- | --- |
| 架构/数据/独立表 | 相同 | 相同 |
| DropPath | 0 | 深度线性0→0.08；前2k关闭、2k–5k渐增 |
| AdamW | 原分组策略、WD0.02 | Conv/Linear权重WD0.05；Norm/Bias/Embedding为0 |
| Batch / LR | 512 / 7e-5 | 同左 |
| Warmup / 调度 | 1500 / 50k | 同左 |
| 第一阶段停止 | 30k | 30k |

DropPath仅作用于残差，不丢shortcut；评测关闭。训练全局step生成设备标量控制ramp，不逐步修改Python模型属性。
X2−X1检验的是整个正则包，不区分DropPath与WD各自贡献。
旧114M的Writer表错误地参与FM且额外10行未独立预训练，因此X1对旧114M不是严格的纯架构单变量对照。

## 入口和验收

```bash
python -m src.unet.train --config src/unet/configs/extreme48m/x1_top20.json
python -m src.unet.train --config src/unet/configs/extreme48m/x2_top20.json
```

启动前需GPU预检B512；不把理论MAC比例当作实际加速。两组控制初始化种子、数据顺序和训练预算。
同口径生成评测继续使用 `tools.cohort_eval.run`；按t及seen/unseen字符诊断：

```bash
python -m tools.cohort_eval.probes.fm --checkpoint CHECKPOINT --out FRESH_OUTPUT --collections top20/validation,top20/train_half
```

FM探针使用posterior mode、相同VAE scale/shift、固定逐图噪声和十个t中点；不是积分后的训练期望loss。
来源列只在明确指定时使用；现有source标签不等于严格作品级留出，不伪造跨作品泛化结论。
保留SSIM、LPIPS、Ink-IoU、Frag、墨量及独立Writer评测，不能只靠训练loss确定更优。

最小验证：表契约及正则化6项CPU测试通过；尚无新架构收敛结果或稳定长训速度。
