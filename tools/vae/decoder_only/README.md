# Decoder-only 受控对照（2026-10-26）

4090 根目录：`/root/Workspace/xy/DiT/assets/decoder_only_20261026/`。
串行入口：`python -m tools.vae.decoder_only.queue`，独占 `/tmp/dit_gpu0.lock`，每个阶段另查 GPU 进程；遇到异常停止整条链，不跳过失败臂。
等待既有 T0/T1 评测结束 → T2 实模前反向检查 → 修正版 T2 5k → V0 2500 → V0 配对评测 → V1 2500 → V1 配对评测。
旧 three_day 调度进程已现场检查为不存在，不恢复旧三日计划。48 的 R1–R4 拆因独立继续。
18:03四项T0/T1补评全部完成；剩余两项改批量后约5.5分钟完成。[补评结果](../../../ANALYSIS/T_group_20261026/README.md)。
18:05修正版T2实模检查通过：同writer MSE对拍、有限非零反向；随后从原P2 EMA正式启动5k。
机器另有 `directuv_surface.prepare_object` 数据准备常驻CUDA上下文，NVML宿主PID2008042、约1098MiB，GPU利用率0%。不杀无关任务；显式白名单上限1536MiB，阶段边界检查新PID/显存/活跃利用率，超限停链。其他训练进程不在白名单。

## 起点与训练

- P2：`assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt`，EMA。
- 从该 checkpoint 的 architecture config 读取 VAE、Top10 CSV、scale/shift，禁止凭默认值替换。
- 实际 VAE `pretrained_models/sd-vae-ft-ema`，scale .18215、shift 0；Top10 `assets/u100_p1p2_20261025/p2_train.csv`。
- V0/V1 都从原始 VAE 独立加载，seed90，B8×accum4，有效32，固定 LR1e-5，AdamW/weight_decay0，保存1250/2500。
- 只训练 `decoder.*`；encoder、quant_conv、post_quant_conv 冻结且 eval。训练用冻结 posterior sample，评测用 posterior mean。
- V0：L1 + LPIPS(VGG)，权重均1；V1 仅增加1/2/4尺度 Sobel L1。
- V1 每个 microbatch 对**全部 decoder 参数**测量基础/边缘梯度范数，将边缘梯度限制在基础的20%以内；系数上限1。不是硬骨架计数损失。
- 第1步/每250步额外记录 L1、LPIPS 各自梯度；不使用重计算或梯度检查点。
- 第1步、保存点检查：非 decoder 状态 SHA256 不变，固定4张 GT 的 posterior mean/logvar 以及 post_quant 输出逐位一致。
- `identity.json`、`contract_check.json`、`metrics.jsonl`、完整 diffusers 权重和 `training_state.pt` 各臂独立保存，不覆盖旧资产。坏图直接报错。
- 原训练入口支持 `python -m tools.vae.train_dino_calli_vae --decoder-only --p2 ... --arm V0 --output ...`；此模式绕开旧全量训练配方。

## T2 修复与边界

旧版除 tau=.05 饱和外，还把 clean z0、未缩放 t 传给错误书家分支；正确分支却使用 noisy zt、t×1000。因此旧 fm_true/fm_wrong 的数量级差异**不能证明书家排名已饱和**。
修正版用同一 `v_target` 重建 `zt=z0+t*v_target`，使用相同时间尺度；dropout 只抽一次，排除被丢弃的 writer/script，错误分支禁止二次 dropout。
tau=.5、weight=.1、t≥.6、B256。错误分支 stop-gradient，仅作为参考误差，避免通过主动恶化错误书家满足排名。
先从原 P2 EMA 实模验证同 writer 重算 MSE 一致、ranking 有图、backward 有限非零，再从同一 P2 EMA正式跑5k，保存2.5k/5k。
修正版单独目录 `fm/T2/`。旧 T2 不延续、不混入结论；B256 与 T0/T1 的 B512 曝光不同，不能称完全等预算对照。
T2训练日志中的`Diff`是FM+.1×ranking的合计，不能直接与T0/T1纯FM数值比较；`T2_RANK`另列active子集的fm_true/fm_wrong。

## 配对验收

复用已存 P2 CFG1 latent bank `assets/p2_audit_20261025/cfg/cfg1.0/`，校验 P2 checkpoint SHA；不为不同 decoder 重采样。
固定100个目标分别做 GT→编码均值→原/新 decoder，以及同一 P2 latent→原/新 decoder。
输出全量500张 PNG、前16目标五列海报、逐图 JSON、L1/LPIPS/PSNR/SSIM、五阈值 Ink-IoU/有效连通块、边缘强度、墨量/笔宽/重心。
按 P2 训练 CSV 单独汇总未见字符和未见 Writer×Script×字组合；这不等于整个通用预训练从未见过该字。样本偏少时明确给 n，不据此宣称泛化已解决。
字符正确性以逐图人工复核为准，未校准 OCR 不包装成真实正确率。相同 STD 下的书家差异方向沿用 T 组共享字 probe，尚不能用单目标平均指标替代。
每次V0/V1评测会自动调用40共享字Writer方向probe；原P2生成latent只创建一次保存在`shared_latents/`，三个decoder复用完全相同的latent与STD，另核验ckpt/CSV/噪声/步数身份。该probe是局部事件诊断，不能当作可靠的端点/分支训练loss。
最终以 P2 生成图是否改善为准，不能因重建指标上涨就判成功。1250保留供早期回溯，2500正式配对验收。

## 48 侧新发现

R2 修正了后验 latent 的重复缩放和灰度/墨迹极性错误；最新20字多阈值修剪分支对照的 VAE/GT 连接数比例约1.09，非先前声称的约7倍。匹配模糊对照约.95。
这些数值来自 `assets/stroke_writer_20261026/R2_blur_control.json`，不能用“匹配模糊解释全部误差”直接证明 decoder 无问题；但旧7倍伪分支证据必须撤回。V0/V1 是受控验证，不是已经确诊 VAE 为唯一瓶颈。
