2026-10-30 00:45跨会话进度汇总：48的S5已到85,000步（train.jsonl实读），187字×3seed终评待确认；4090侧00:26后SSH入口超时，S0以19:22核验的56.7k/85k为准。本轮matrix64旧计划50k矩阵8臂全部完成评测（S0 0.6027最高、S8 0.6001次之、全距0.016），J0–J3延长按10-28 22:28第二代方案有意终止于J0@11,654。48关键权重18.6GiB已于10-28 12:42完整备份至本地。以下各条目保留。

# 训练进度

2026-11-01 22:34（**GARL 4h 长跑启动 + GARU 解冻臂实现完成**，4090 tmux `garl`）：
按用户 12h 预算「先让 GAR 跑 4h，中间 eval 观测，同时做好解冻实现」。

**(a) GARL 长跑**：spec `assets/d_20261101/specs/GARL.json`（make_gar.py 一并生成）。
与 GAR **同一实验、两个预算**：同起点 K、同冻结、同目标函数，只改 LR horizon 与 ckpt 间隔。
- 4000 步（**11.93 epoch**），`ckpt_every=eval_every=250`（16 个 ckpt × 1.71GB ≈ 27GB，
  磁盘 182G 余量够），实测 0.285 step/s，ETA **≈3.97h**（约 02:35 完成）。
- ⚠️ **GAR 不是 GARL 的前缀**：672 步之后两者 cosine 日程 horizon 不同、LR 路径分叉，
  **不可互相 resume**；但同步数下可比。
- 启动即复现 GAR 第 1 步损失（mse 0.007525 / lpips 0.031836 / DINO 0.490945 /
  gram 0.001214），确认两条线同源。

**(b) GARU 解冻主模型——设计与实现已就位，尚未训练**：
- 起因：D2 **其实已经跑过联训**（`joint.arm=J3`），当时结论「联训伤了 Writer」
  （direction 0.4291）**已作废**——`build_specs.py` 剥掉了 `transfer_style`，
  导致 D1/D2/D3 全是「K 主干 + **随机** style 分支」（33/368 张量不同，
  max|diff|=36.06 在 `style_injections.2.gain`）。所以 **联训至今未被干净测过**。
- `joint.arm=GARU`：同 GAR 目标，改 `freeze_backbone=False` + `skip_fm=False`
  + 保留生成器图（`detach_image=False`）；**显式补回** `transfer_style=True`、
  `calibrate_gains=False`（否则 `condition_report(calibrate=True)` 会把 K 的
  gain 14.81/24.27/37.06 重标成 6.01/19.87/22.52）；`lr_backbone=5e-6` 沿用 D2。
- **验收口径必须换**：不跑 frozen-K 审计（生成器本就该动），改为与 K 的**配对比较**；
  此时 direction 反而是有效读数。
- 两道门已建：① CPU 构造预检 `check_spec_init.py` 实测 GARU
  **368 tensors max|diff|=0.000e+00**（起点确实是 K）+ **trainable 362/365**（确实解冻）
  ——这正是能提前抓住 D2 那个 bug 的检查；② 梯度路由探针 CPU 侧已验证图策略
  （GAR/G2J4 阻断、GARU 保留），GPU 侧数值待 GARL 让出显卡后补跑。
- `tools/d_20261101/run_garu.sh` 会先等 GARL 结束，再依次过两道门并训练 2500 步（≈2.5h）。

2026-11-01 21:19（**GAR 解码器后训练启动并跑完 2 epoch，4090 tmux `gar`**）：
按用户要求「不动 CFG、保持 K 的原貌、复用已有实现」，把既有 GAR 目标接进仓库审计过的
joint 训练链路（**没有**另写训练脚本），只训练解码器。spec
`assets/d_20261101/specs/GAR.json`（由 D1b 派生，生成器 `tools/d_20261101/make_gar.py`）。

- **CFG 保持 1.0**：生成器只在 `condition_dropped=True` + `cfg=1.0` 下调用，与 K 部署口径一致。
- **K 原貌审计通过**：step 100 checkpoint 对 K 逐张量比对 `368/368 tensors, max|diff|=0.000e+00`
  → `delta` 与 `ema` 均 IDENTICAL，解码器 138 张量确实在动（total norm 2434.82）。
- **只改两处**：`joint.arm=G2J4→GAR`（目标函数换成 MSE+LPIPS/DINO+10×Gram，**无 edge 项**）、
  `decoder_lr 5e-6→5e-5`（5e-6 在 1000 步只动 ~1e-3，等于没动；本线只跑 2 epoch，必须真能移动）。
  另 `compile=False`（`skip_fm` 下生成器不编译）。
- **曝光 672 步 = 2 epoch**（21460 unique 作品，`image_batch=64` → 1 epoch = 336 步），
  实测 **0.287 it/s（3.48 s/step）**，全程 ~39 min，peak 19.28 GiB / 24.5 GiB。
- 启用前提先修掉一个真 bug：旧独立脚本 `tools/d_20261101/gar_train.py` 直接把 CSV 的
  `script_id∈{0,3,4}` 喂给只有 4 行（3 字体+null）的 `backbone.script.weight`，越界触发
  `CUDA error: device-side assert triggered`（**不是** cuDNN 故障）。正式口径必须经
  `font_remap.json→{0,1,2}`、`callig_remap_p125.json→0..46`，且 STD 条件 g 用 σ=0.65 软字形、
  latent 用 `shards_all` 缓存（53522/53522）。该独立脚本已废弃。
- 训练损失（GAR 预期特征：纹理/感知降，pixel MSE 平）：step 1→270，DINO 0.494→**0.244**、
  Gram 0.001212→**0.000694**，MSE 0.00752→0.00826 基本持平。
- **验收六门全过**（对照 D1b，其生成器与 K 逐位相同）：SSIM +0.002288、LPIPS −0.002946
  （两者 100→336→672 **单调**改善）、Ink-IoU **+0.000398**（门是跌幅≤0.003）、OCR 77.78→77.78
  持平（n=90）、w50 仅 +0.28%（无变粗漂移）、writer direction 恒 0.5571（生成器未动）。
  改善量级很小（SSIM 相对 +0.38%）——这是解码器后训练的物理上限，latent 由冻结的 K 决定。
- 面板评测 `assets/d_20261101/eval/GAR_s*`（val100，`--use-ckpt-decoder`）；报告
  [GAR_DECODER.md](docs/1031/GAR_DECODER.md)。

2026-11-01 18:50（**D1c 启动：从 K 正确起点的 6h decoder 长训练，4090 tmux `d1c`**）：
按用户要求，从当前最佳权重 **K**（`assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt`）
重新做 **decoder-only 长训练，ETA 6h**。spec `assets/d_20261101/specs/D1c.json`（由 D1b 派生，
生成器 `make_d1c.py`）。
- **与 D1b 完全同源同法**：`transfer_style=True`、`freeze_backbone=True`、`skip_fm=True`、
  `decoder_lr=5e-06`、`lr_backbone/lr_style_inj/lr_style_repr=0`；
  实测 `grad_norm=0.0`（生成器零梯度）、`decoder_grad_norm 0.35–3.39`（只有 decoder 在动）。
- **只改三处**：`max_steps` 2500→**7500**、`lr_decay_steps` 2500→**7500**（原来 2500 步后
  LR 就压到 min_lr，6h 里大半时间等于不学）、`ckpt_every`/`eval_every` 500→**1000**
  （每 ckpt 1.71GB，减半省 12GB 磁盘；磁盘余量 183G）。
- **不是从 D1b 续跑**，而是从 K 重新起跑，符合「正确的起点长训练」。
- 实测步速 **0.33 step/s**（0.329–0.332 @ step 70–90），7500 步 ≈ **6.3h**，
  与 ETA 6h 吻合。peak 18.29 GiB / 24.5 GiB。
- 启动踩到两个坑已修：① 旧入口 `tools.d_20261101.train` 在远端缺 `common.py`，
  正确入口是 `tools.style_matrix.runtime.train`；② 共享 trainer 断言
  `eval_every == ckpt_every`，只改 ckpt_every 会直接拒绝启动。
- 日志 `assets/d_20261101/logs/train_D1c.log`，权重 `assets/d_20261101/execution/runs/D1c/checkpoints/`。
- **D1b 结论不变**：2500 步时生成器 368/368 与 K 逐位相同，E-G 75.94 = K，OCR 77.78。
  D1c 用来回答「decoder 训更久会不会突破」——预期看 SSIM/LPIPS 是否延续 500→2500 的单调改善。

2026-11-01 18:28（**MoYi GT 标签反推成功，E-G×OCR 已入图**）：按用户要求继续攻 MoYi 标签。
**哈希路全灭**：file/RGB/L/反色/8级量化 5 种哈希，对 26,768 张图索引
（eval200_fixed 187 + eval200 187 + combo_unseen 112 + 主索引 9000 + candidate 8282 + noaug 9000）
**0/187 命中** —— MoYi 的 GT 不在本地任何图库。
**改走「GT 自解码 + 形状一一分配」成功**：
① 项目 OCR 直接读 187 张 GT → 184 唯一字、平均置信 0.613；与 eval200_fixed 位置匹配 137/187，
50 个分歧中 38 个置信<0.5（`復/复`、`彌/弥`、`數/数` 等异体误读）。
② 二值化形状跑匈牙利最优分配 → **163/187 落在同一行**，书家覆盖全 10 位（14–26 张，分布合理）。
③ 用解码字自评 GT 得 **100%（187/187）**，证明标签自洽。
据此把 eval200_fixed 的字+书家迁移给 MoYi 187 行，E-G 与 OCR 均可算：
**MoYi 12ch@50k = E-G 75.18 / OCR 55.23**（此前基于错配协议的 77.82/63.37 作废）。
**E-G×OCR 图已放入 MoYi 12ch（按用户决定 4ch 不放入该图）**，同图并补入 L-prior 190k/256k
（std prior only，无字表）。SSIM×LPIPS 图中 MoYi 12ch/4ch 均保留（该图不需标签）。
CSV 现 151 行。

2026-11-01 18:22（**MoYi GT 反推成功一半**）：按用户要求对 MoYi 做 GT 哈希反推。
**哈希路全灭**：file/RGB/L/反色/8级量化 5 种哈希，对 26,768 张图索引（eval200_fixed + eval200 +
combo_unseen + 主索引 9000 + candidate 8282 + noaug 9000）**0/187 命中** —— MoYi 的 GT 不在本地任何图库中。
**但 OCR 路走通**：MoYi 的 GT 是正常渲染（白底黑字、墨占比 7–15%），直接用项目 OCR 解码得到 187 个字、
184 个不同字、平均置信 0.613；与 eval200_fixed 规范顺序位置匹配 137/187，分歧是 `復/复`、`咏/瓏` 繁简异体互读。
用解码字作标签后 **GT 自评 100%（187/187）**，证明标签自洽；**MoYi 12ch 生成图对该标签 = 48.7%（91/187）**。
此前基于错误协议算出的 63.37% 作废。
**SSIM×LPIPS 中 MoYi 已恢复**（该图不需标签，生成图对自带 GT 原生合规）：12ch 0.5788/0.4318、4ch 0.6008/0.3970。
**E-G 轴仍不可用**：解码只能得字、得不了书家，书家标签无法恢复。

2026-11-01 18:15（**SSIM×LPIPS 补齐 + MoYi 剔除**）：新增 SSIM×LPIPS 全代际图（43 面板，全部重算，
非抄旧报告）。**v80@190k 双第一：SSIM 0.6192 / LPIPS 0.3805**；scan 系列 20k→200k 呈明显台阶
（0.5374→0.6187）。SSIM 与 LPIPS 在这批数据上近似单调一致（同一维度），
与 E-G/OCR 的双轴分离形成对照。
**MoYi 两个点已按用户判定从全部图与表中移除**：其自带 GT 与 `eval200_fixed` 不同批
（最近邻仅 111 唯一行、均差 0.096）、原始协议 CSV 已丢失，用协议补 GT 只能证「顺序」不能证「配对」，不可比。
E-G×OCR 图现为 132 点。

2026-11-01 18:05（**全代际 E-G × OCR 完成：132 个身份可证面板**）：历史面板全部补齐并重测。
关键突破：37 个「只有生成图」的面板，其协议就是 `exp-std/csv/eval200_fixed.csv`（187 行图像本地全可读），
**从协议渲染 GT 后核验全部 exact=True**，零传输补齐 27 个。MoYi 12ch 自带 GT 与协议不符
（最近邻仅 111 个唯一行、均差 0.096），改用协议 GT 后确认其生成图属同一协议，成功重测。
**MoYi 4ch @80k 是历史最强：E-G 85.58 / OCR 83.14（全图 OCR 最高）**；MoYi 12ch @50k 仅 77.82 / 63.37。
CFG 扫描呈对抗：1.2→1.6 使 E-G 86.74→89.93（最高）但 OCR 66.28→65.12。
全图无模型同时占据两轴顶点，必须双轴并列。图 `assets/metrics_index/fig_pooled_eg_ocr_final.png`，
表 152 行 `paired_eg_ocr_20261101.csv`，盘点 `docs/1031/HISTORICAL_PANELS.md`。

2026-11-01 17:50（**D1b 完成 + 统一重测收口**）：D1b 已跑满 **2500/2500 步**，末次记录 0.3507 step/s、
grad_norm 0.0、decoder_grad_norm 0.23，checkpoint `assets/d_20261101/execution/runs/D1b/checkpoints/0002500.pt`
（1.7GB，16:55 落盘，训练进程已退出）。`verify_d1b.py` 断言 **368/368 生成器张量与 K 逐位相同**（max|diff|=0），
decoder 138 张量 norm 2434.8 —— 证明只有 decoder 被训练。正式评测（`tools/f_20261101/eval_arm.py`，
val100 100 字）：SSIM 0.5946 / LPIPS 0.3462 / Ink-IoU 0.2886 / 碎片比 2.658，OCR top1 **77.78**（n=90），
pooled E-G **75.94 / EER 30.58**，与 K 的 75.94 完全一致。

统一重测已完成：**107 个身份可证面板**（远端 121 + 本地历史 19 历史点，剔除诊断臂）同时具备
official pooled E-G 与同面板 OCR。口径用旧冻结缓存复现 S4 = 84.908（发布值 84.91）验证。
E-G 特征抽取改在 4090 GPU（D1b 释放后），121 面板数分钟完成；历史 19 面板在本地 4070 上补跑
（`assets/metrics_index/eval_historical_local.py`，复用远端 3461 行参考库）。图与全量表：
`assets/metrics_index/fig_pooled_eg_ocr_final.png`、`paired_eg_ocr_20261101.csv`；
存档盘点 `docs/1031/HISTORICAL_PANELS.md`。旧 45 面板 retrieval/AFC 表已作废，不再引用。

2026-11-01 重测接续核查（**已被 17:50 条目取代**）：D1b 已实读到 2200/2500，0.3502 step/s，生成器 grad_norm=0、decoder_grad_norm=0.618；未增加训练预算。E-G/OCR 统一重测正在 CPU 执行。此前“官方 pooled AUC 实现消失”的判断错误：正式实现仍在 `tools/eval_unpaired/domain/statistics.py`，不可将 retrieval AFC 当成 AUC。OCR 批处理 str/Path 错误已修并重试；面板清单的 step=0 与跨步去重缺陷正在纠正，旧45面板结果不可直接作为最终全局图。

2026-11-01 14:05（**D1b 拉起：修正起点后的 decoder-only，4090 tmux `d1b`**）：复查发现
D1/D2/D3 三臂**从未从 K 出发** —— `build_specs.py` 剥掉了 `transfer_style`，而
`common.py:34-46` 只在 `transfer_style` 为真时才复制 `style_representation` / `style_injections`，
于是三臂都是「K 的主干 + 随机初始化的 style 分支」。三个 injection gain 实测为
59.11/66.84/63.86（K 是 14.81/24.27/37.06），且三臂完全相同（`construct()` 用 `config.seed`
确定性初始化），这就是指纹。**不是 EMA 漂移**：`train.py:198` 冻结时跳过 `update_ema`，
实测 `ema_D1 == raw_D1`（差 0.0）。`check_init.py` 证明：当前 spec 与 K 有 33/368 个张量不同
（max 36.06 @ `style_injections.2.gain`），加 `transfer_style=True` 后 368/368 全等。
该混淆推翻了首轮结论 —— 用 `--decoder-from` 把 D1 的 decoder 挂到 K 上，SSIM +0.0036
（96/100 胜）、LPIPS +0.0047（90/100）、Ink-SSIM +0.0033、Ink-IoU +0.0007、碎片比 +0.2813
五项显著改善，只有容差骨架 F1 −0.0005 显著退化；且 K+D1dec 每项都优于 D3+D1dec，
说明 decoder 从未在 K 上退化。decoder 步数扫描（K 生成器 × D1 的 500/1000/1500/2000/2500）
单调改善且 1500 步已得约 90%，面板 OCR 全程 77.78 不动，骨架 F1 单调缓降 0.0258→0.0253。
**D1b** 从 4090 上已调好的 D1.json 派生（保留 `image_batch 64 / image_micro 8`），
仅改 `transfer_style=True` + 身份，2500 步约 2h。启动即核验：`transfer_style: 34`、
`strict_backbone_equal: True`、`frozen 362 generator tensors`、`grad_norm 0.0`、
`decoder_grad 0.44–3.40`、19.8 GiB / 95% / 416 W。首 checkpoint（500 步）落盘后用
`verify_d1b.py` 断言生成器与 K **逐位相同**。详见 [docs/d_results.md](docs/d_results.md)。

2026-11-01 11:15（**D 计划首轮完成，4090 已空闲**）〔**已被 14:05 条目推翻**：三臂起点错误，结论不成立〕：D1/D2/D3 串行跑完并全部评测，
外加不额外训练的交叉组合 D3+D1dec。实测步速与 ETA 与预估一致：D1 04:30→06:29（2500 步）、
D2 06:29→08:45（3000 步）、D3 08:45→10:44（11000 步）、评测 10:44→10:59。**结论：K 仍最优，
D 计划没有产出可替换它的权重**——配对检验（100 张同图同种子，bootstrap 1e4）里只有退化达到显著：
D1 的 LPIPS −0.0100 与 Ink-IoU −0.0256、D2 的 Ink-IoU −0.0132；D3 全部 n.s.；
D3+D1dec 观感最好（SSIM 0.5965 / LPIPS 0.3428 / OCR top5 98.44 / 碎片比 2.389，均为四臂最优）
但 CI 全部含 0。**机制**：图像域损失奖励平滑高墨量渲染，decoder 靠加粗+合并笔画把碎片比压下去
（端点密度 0.0008→0.0004、连通块 10.16→8.28、笔宽 0.970→1.004 与 GT 齐平），
所以碎片比下降是假改善、粘连加重，Ink-IoU 与 OCR 同步下滑；D1 单独使用时 Writer 身份崩塌
（direction 0.5454→0.1607）。D3 是唯一不伤 Writer 身份的臂（0.5372/0.653）。
**不建议据此追加 2–4h**，增益在 100 字面板上与噪声不可分。另修一处评测缺陷：
`t3_writer_swap` 原先不认 `--decoder-from`，交叉组合的 t3 报告与 D3 md5 完全相同；
direction/amplitude 在 latent 空间算本就不受影响，但 OCR 走图像，原先拿的是 D3 的原样结果，
补跑后为 96.22/98.44。结果表与判据见 [docs/d_results.md](docs/d_results.md)，对照图见
`assets/d_20261101/review/`。

2026-11-01 04:35（**D 计划拉起，4090，tmux `d` 串行 D1→D2→D3**）：按用户裁定停掉 F 计划（F0 未产出 ckpt），改做三类缺陷对应的三个后训练臂，全部从 K（`0043429.pt` EMA）出发、保留完整 Writer 注入、冻结 VAE Encoder、不新增条件网络。实现落 `tools/d_20261101/`（`calibrate.py` / `hard_chars.py` / `build_specs.py` / `tune.py` / `measure.sh` / `show_spec.py` / `run_d.sh`），并改三处运行时：① `runtime/common.py` 的 `optimizer()` 在 style-matrix 分支补上 VAE decoder 参数组（**此前 decoder 静默不训练**，`lr_decoder` 缺省 5e-6）；② `step()` 新增 `fm=False` 路径（跳过速度回归但仍消耗同样 RNG，D1 用）；③ `runtime/train.py` 支持 `freeze_backbone` / `skip_fm` / `decoder_lr`，冻结生成器时跳过 `gradient_report`（否则「style 分支必须有梯度」断言必炸），并把 `steps_per_s` 计时改为带 `torch.cuda.synchronize()` 的 10 步粒度记录（**原先 CUDA 异步导致步速虚高数十倍**）。**实测（nvidia-smi，40 步，单臂独占卡）**：D1 2.83 s/step、峰值 19.8 GiB；D2 2.60 s/step、21.2 GiB；D3 0.64 s/step、22.4 GiB。horizon 由此反推而非预设：D1 2500 步、D2 3000 步、D3 11000 步，各约 2.0–2.2 h。D1 用 **G2J4**（image-only、生成 latent、**detach 生成器**、只训 decoder），D2 用 **J3**（FM + 弱图像 + GT 重建锚），D3 为纯 FM + 困难字过采样（`stage_p2_hard.csv`，1.13×，难度取自**训练集自身**的 `endpoint_density/fragment_share/width_p50/ink_ratio` + 频率，验证集不参与）。坑：`eval_every` 必须等于 `ckpt_every`；卡上残留进程会污染显存测量（曾出现 3 个进程共占 9.8 GiB 导致 D3 假 OOM）。D1 当前 19.8 GiB / 95% / 413 W，ETA ~2.2 h。设计、验收门与判据见 [1101.md](1101.md)。

2026-11-01 01:10（F 计划启动，4090 GPU + tmux 串行）：按诊断结论实现并启动 **12h 单方向后训练**，只做「摘掉有害注入 + 强化 8×8 书家通路」，不碰架构/VAE/主干。实现落 `tools/f_20261101/`（`build_specs.py` / `eval_arm.py` / `pick.py` / `run_f.sh` / README），并改三处运行时：① `runtime/common.py` 新增 `disable_sites/assert_sites_disabled`（gain 是 buffer，`delta*gain` 置零即精确关闭）；② `runtime/train.py` 支持 spec 字段 `disable_injection_sites`，在 raw 与 EMA **各自 load 之后**分别置零+冻结，存盘前强制断言，checkpoint 记录 `disabled_injection_sites`；③ `tools/diag/t3_writer_swap.py` 支持 `--off-sites` / `--gain-scale` / `--cfg`；④ `tools/p125/eval_cpu.py` 支持 `--device cuda`。**冒烟全绿**：60 步→resume 80 步后 `style_injections.0/1.gain` 在 delta 与 ema 中均为 **0.0**、存活 site2 gain 保持 21.6264；四组 LR 比例 5e-6:5e-5:8e-5:5e-5 正确；F1 的 E-G 实测 **eg_samples=4、eg_calibrated=true、achieved ratio 0.0200**（target 0.02，不再像 P3B 静默空转）；`eval_arm` 端到端出 content+OCR+笔宽+writer 四轴（val100: ssim 0.5798 / OCR top1 74.44 / width_p50 7.61）。串行链：s0 K 选择（A@43429 vs P3A@20000）→ s1 四消融 → s2 spec → s3 冒烟 → s4 F0 5k → s5 F1 5k → s6 选优 → s7 赢家 +7k 至 12k → s8 终验+CFG/gain 扫描；tmux `f`，日志 `assets/f_20261101/logs/f_driver.log`，阶段幂等（`state/*.done`）。训练侧前置：abc 链已在 43,429 步 SIGTERM 并落盘完整 `0043429.pt`，GPU 全空。计划与验收门见 [tools/f_20261101/README.md](tools/f_20261101/README.md)。

2026-11-01 00:35（P125 第二阶段机制诊断定稿，4090 GPU）：**四个无训练诊断已在 GPU 上用正确 Top10 数据（`stage_p2.csv`，writer 槽 0–9）重跑完成**。**先更正上一版结论**：10-31 23:45 的 CPU 版取样槽位错误（用了 `p125_trusted_encode.csv` 的通用槽 10–46，而 P1/P2 只训 Top10），该版“书家条件失效”结论**作废**。契约审计（决定性）：stage_p0=10–47（通用）、stage_p1=0–9+47、stage_p2=**0–9 only** 且一对一零混槽；`backbone.writer` 冻结（P0→P3A 漂移 0.0000），真正可训练的 `style_representation.table` **rows 0–9 漂移 0.1862、rows 10–46 仅 0.0017** —— 学到东西的正是 Top10 那 10 行，**训练契约没有错位**。重跑结果：T1（60 行）ΔL_wrong 仅 **+0.0012**、ΔL_null **−0.0289**（空条件更低是单样本 MSE 的必然，不是条件有害）；T2 关掉 32×32/16×16 后 FM loss **下降 0.0154/0.0085**（净负担），关掉 8×8 则书家区分度只剩 **retention 0.334**；T3（900 张，GPU 1.8 min）direction **0.4082**、amplitude **0.5868**、OCR top1 **96.11%** —— 书家条件**确实接通但弱**；site 消融扫描证实书家身份**几乎全由 8×8/576ch 承载**（关掉后 direction 0.3595→**0.1659**），摘除 32×32/16×16 对生成指标无损；T4（300 张）OCR 100%→100%（drop 0.0）、SSIM 0.97405、Ink-IoU 0.97987 ⇒ **VAE 不是瓶颈**。第三阶段单方向定为 **I 注入结构整改**（摘除/门控 32×32+16×16，强化 8×8 书家通路，writer-table 预热 3–5k 后解冻中后段 10–15k），验收门：8×8 retention ≥0.6、direction ≥0.6、amplitude∈[0.8,1.2]、OCR 未见字不退化。训练侧已按用户指令停止：abc 链 A@35k→45k 在 43,429 步 SIGTERM，落盘完整 `0043429.pt`，GPU 已释放，serial 编排已硬停不再重启。报告见 [docs/1031/p125_phase2_diag.md](docs/1031/p125_phase2_diag.md)。

2026-10-31 22:20（P125 S5 链收口，4090）：**P125 全六段完成**（本会话监控期 22:2x 收刀→02:1x P2 完成；02:08 会话失联 20h，期间接管方跑完全链并启动新计划）。heldout 终局轨迹（val100 n=100，同种子行）：P0@50k ssim 0.5573 / P1@5k 0.5636（frag 10.69→2.82，**−74%**） / P2@15k **0.5857**（ink_iou 0.2723，单段最大增量 +0.022） / P3A@20k 0.5857 / P3B@20k **0.5881**（frag 2.64、hole 0.0437 最优）。**EG pilot 门 failed**：P3B spec 的 EG 形态参数未进 resolved_config（仅 pair_fraction 0.3，无 eg/joint 键）→ runtime `eg_skipped=true`（eg_samples 0、decoder_grad_norm 0）静默空转 → probe 无辅助行判 fail；P3B 实为纯 FM 对照（差值 +0.0024 在种子噪声内）。EG 机制待修再验证。4090 GPU 已由**另一会话的 abc_20261031 计划**接管（A@25→35→45k、B@75k、C1@55k、C2@75k，当前 A@35k resume 中），P125 资产只读不再动作。权重登记见 [ckpts.md](ckpts.md)。

2026-10-30 22:05（P125 S5 链，4090）：P0 heldout 全轨迹收齐（val100 n=100，CPU eval 内核，每点 460s 不停训）：1k 0.263 / 5k 0.533 / 10k 0.549 / 20k 0.557 / 30k 0.560 / 40k 0.557（ssim），**自 ~20k 平台化**，训练 fm 同期 −13% 为纯 seen 增益。用户裁定：**P0 收刀于 50k**（编排已挂，50k watcher eval 平台门 ssim≥0.55 通过后自动 SIGTERM→改 plan stop→重 seal→重启链→P1）；裁定**不补 S4@85k**；转入全程目标监控（goal-e8401d41 + 45 分钟巡检）确保 P1→P2→EG pilot 门→P3A/P3B 正确充分完成。计划变更细节见[1030p125.md §2.5.4](1030p125.md)。当前 47.8k 步，1.553 step/s 恒定，peak 20.41GiB。

2026-10-30 13:21（P125 S5 链，4090）：P125 全链拉起（run_p125.sh，tmux p125）：batch512/microbatch512/compile=True，实测稳态 **1.554 step/s**（0.644 s/step），peak **20.41 GiB**（+1.1GiB 容器外常驻）。此前修复空白：`src/unet/io.py` 未传 num_writers（P0 style 分支永久零梯度根因）、microbatch 256→512、P2/P3 source、阶段间 style 权重搬运（transfer_style）、init_p1_table 语法错、build_stage_csvs NameError、新增 seal_ready。SSH 直连 4090（ProxyJump 48 已注释）。

2026-10-29 20:28文档汇总：本次两端SSH重连均被10.222.120.101入口关闭，不能刷新现场进度；未停止/重启训练。最后成功核验仍为19:22的S0 56.7k、S5 56.75k，队列计划不变。全部明文指标及历史实验见[完整报告](docs/1029/full_report/README.md)。

2026-10-29 19:22：S0 56,700/85k、S5 56,750/85k；S4排在S0之后。累计正式速度约1.597/1.660step/s，训练allocated峰值20.02/21.38GiB。S5剩约4.7h；4090的S0剩约4.9h，后接S4约6.4h，共约11.3h，评测另计。55k原子checkpoint及.done均存在。HWD 30面板与跨字核验、CHAED探索回归校准均已完成，CPU执行，未停训练。结果入口[10-29汇总](docs/1029/README.md)。下方是历史快照。

2026-10-29 18:17：新三模型85k对照已启动（4090 S0→S4，48 S5）；旧扩展链保持暂停。最新计划见[focused85k](docs/1029/style_focus_85k/README.md)。以下早前快照保留作历史。

2026-10-29 17:01：S11已完成50k及新指标，双机评测已收口；训练自动队列继续暂停。最新结论见[新评价专题](docs/1029/eval_unpaired/REPORT.md)，下方早期快照保留作历史。

10-29本轮64h队列只读核查：4090 S9到45,650，48 S11到15,050；主控制器及两端later_stage观察器均存活。核心成熟/独立seed链已挂，匹配LR新版等待S9Pair@60k、S11@50k边界激活。H0/H1联合路线未加入自动队列，不能称附件全部候选均已挂齐。

2026-10-29 09:17：4090 S9约44.8k，48 S11约14.15k，训练持续推进。48/64h新版仅暂存：4090等待S9Pair@60k机制评测完成，48等待当前S11@50k机制评测完成，再在GPU空闲边界接续；未停止或修改活动训练。交接登记见各根`later_handoff.json`，实现说明见docs/1028/later_stage.md。

2026-10-29 09:08只读快照：4090 S9到44,150/50k，累计1.239step/s；48已完成J4–J7@5k，decoder_gate选择为空，已自动转入S11冷启动，GPU100%。阶段报告固定使用10-28 21:30截面，新训练状态不回填历史结论；本次文档归档未修改活动源码或训练进程。

后台确认（2026-10-28 23:20）：双机24h队列已运行。4090 S9到500步、GPU100%，后续从同一50k checkpoint依次分叉S9-FM/S9-Pair各到60k；48 J4到1,700步、GPU91%，后续J5→J6→J7各5k。控制器分别为PID887571/3463080，均nohup后台运行。新根为`assets/style_matrix_20261028/generation2full/{4090,48}`；暂停其他实现改动，不再重启当前训练。

最新切换（2026-10-28 22:28）：按[第二代方案](docs/1028/generation2.md)停止旧矩阵延长。48的J0安全停于11,654并完整保存；新J4真实5步通过，J5条件Decoder真实短测及统一评测进行中，通过后自动串行J4–J7各5k。4090的S8约47k，完成50k与评测后接续S9本卡预检及50k冷启动。下方21:30旧队列快照保留作历史，后续安排由本段替代。

更新时间：2026-10-28 21:30（Asia/Shanghai），以下为双机日志与评测快照。完整结果聚合见 [docs/1028](docs/1028/README.md)，计划见 [1027.md](1027.md)，权重见 [ckpts.md](ckpts.md)，资产和空间见 [Data.md](Data.md)、[Disk.md](Disk.md)。

| 机器 | 当前任务 | 步数 / 目标 | 状态 | 速度 | GPU |
|---|---|---|---|---|---|
| 4090 24GB | S8，Local CNN / DynamicConv | 40,400 / 50,000（80.8%） | 正常训练 | 约1.56 step/s | 峰值allocated 20.38 GiB |
| 48GB | J0，5k→12k延长 | 5,150+ / 12,000 | scheduler兼容修复后正常恢复 | 预热后回升中 | 约20.7 GiB，GPU 98% |

速度取正式训练窗口，累计速度仍包含少量启动、保存开销。4090此前S0停在1,350，从1k恢复后已完成50k；当时futex等待的根因未定位，不应称为已明确修复。

## 队列与预计时间

| 机器 | 已完成 | 当前与后续 | 估算 |
|---|---|---|---|
| 4090 | S0/S1/S4/S6各50k并完成评测 | S8（40.4k）→ 等待成对75k选择 | S8训练余约1.7h，另加评测；成对延长另需约9h |
| 48 | S2/S3/S5/S7各50k及评测；J0–J3各5k | J0→J1→J2→J3分别延长至12k | 恢复后按原实测速率估约10h训练+评测，约10-29上午完成 |

4090在五组S完成50k后，脚本等待`selection.json`选一对继续到75k；该阶段不自动挑选模型。48的J延长首次恢复因checkpoint scheduler使用`_last_lr`、代码只读`last_lr`而中止；现已兼容两个键名、校验优化器参数组数量、更新封存源码哈希并从原5k checkpoint恢复。没有跳过身份或配对校验。

## J组Loss进展

J0–J3训练FM末千步日志均值均约0.0402，首千步约0.0525；J2/J3 Image L1从约0.0157下降到0.0106，最后两个千步窗口趋平。图像日志是最后8张microbatch的分量，反向仍累计16张。仅凭训练Loss和唯一的5k固定集评测无法确定收敛步数；继续既定四组12k对照，建议在10k补评。详细窗口与条件见[J组Loss及磁盘审计](ANALYSIS/style_repr_injection_matrix_20261026/j_loss_disk_20261027.md)。

## 已完成的评测

| 模型 | 评测 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ |
|---|---|---:|---:|---:|---:|
| 4090 S0@50k | 100字，EMA | 0.60275 | 0.33950 | 0.28587 | 2.627 |
| 4090 S1@50k | 100字，EMA | 0.58659 | 0.34940 | 0.28062 | 2.265 |
| 4090 S4@50k | 100字，EMA | 0.59875 | 0.34055 | 0.29513 | 2.299 |
| 48 S2@50k | 100字，EMA | 0.59494 | 0.34554 | 0.28350 | 2.489 |
| 48 S3@50k | 100字，EMA | 0.59756 | 0.34258 | 0.29036 | 2.601 |
| 48 S5@50k | 100字，EMA | 0.59750 | 0.34246 | 0.28509 | 2.385 |
| 4090 S6@50k | 100字，EMA | 0.59796 | 0.34284 | 0.29009 | 2.399 |
| 48 S7@50k | 100字，EMA | 0.59717 | 0.34311 | 0.28572 | 2.477 |
| 48 J0@5k | 187字，EMA | 0.61488 | 0.32198 | 0.34373 | 2.397 |
| 48 J1@5k | 187字，EMA | 0.61494 | 0.32220 | 0.34280 | 2.389 |
| 48 J2@5k | 187字，EMA＋配套Decoder EMA | 0.61977 | 0.31505 | 0.34468 | 2.236 |
| 48 J3@5k | 187字，EMA＋配套Decoder EMA | 0.61934 | 0.31474 | 0.34328 | 2.186 |

J2/J3相较J0/J1有初步SSIM和LPIPS收益，J2与J3差距很小；尚需同协议多seed、实际读图、交叉Decoder和12k评测，无法确认重建锚是否带来稳定收益。S0为不同测试集及训练起点，不能与J组直接排名。

逐样本bootstrap、严重失败率、笔宽/结体/重心、BSD、三阈值2px容差骨架F1、独立Writer及S/J海报已统一聚合到[docs/1028](docs/1028/README.md)。S7补齐后没有超过同表征S3或同注入S6；DynamicConv仍未显示稳定内容/拓扑/书风优势。J2/J3的SSIM/LPIPS增益可靠，但骨架F1不升且彼此不可区分。

## S组当前对照结论

S0是从零训练的`Vector + Global AdaGN`基线，也是目前唯一完成50k正式评测的S模型。S0@50k：SSIM 0.60275、LPIPS 0.33950、Ink-IoU 0.28587、Frag 2.627、SoftInk 1.0249、near-blank 0%。实际海报显示已学会清晰的书法化映射且整体墨量接近GT，但仍有局部粘连、缺/错笔、过粗墨块和结体漂移。

`Skel-IoU=0.0253`是1像素骨架的精确位置IoU，不能直接解释为“拓扑只有2.5%”。阈值0.3–0.7复算均约0.025，排除二值化阈值错误；允许1/2/3像素容差后的对称骨架F1为0.124/0.191/0.256，说明主要问题是精确中心线对位过敏。同一`validation_val100.csv`下，P1@10k/P2@20k/通用预训练FT@25k分别只有0.0156/0.0186/0.0179，S0反而更高。历史Deep-Sym4的0.039–0.045来自另一套187字CSV；0.3–0.9历史值多为显式骨架条件或VAE同图往返，均不可直接比较。当前输出的set标签仍写`eval200fix`但实际CSV是`validation_val100.csv`，属于命名/报告问题。复算证据见[S0骨架指标审计](ANALYSIS/style_repr_injection_matrix_20261026/measurements/verification/s0_skel_metric_audit.json)。

S1仅把注入从Global AdaGN改为空间AdaGN，50k时除Frag外整体低于S0；海报还显示更重的墨量与局部墨块，说明当前Spatial AdaGN并非免费收益。S4（Writer×Local CNN + Spatial AdaGN）恢复到SSIM 0.59875、LPIPS 0.34055，并把Ink-IoU提高到0.29513、Frag降到2.299，是目前新结构中最均衡的一组；但仍未全面超过S0。S7（Multi-Token×DynamicConv）为SSIM 0.59717、LPIPS 0.34311、Ink-IoU 0.28572、Frag 2.477、Writer Top-1 58%，没有超过同表征S3或同注入S6。当前证据支持“局部CNN×Spatial AdaGN”这一交互，而不支持Cross-Attention或DynamicConv本身具有稳定优势。等待S8补齐最后交叉项。

## 路径与运行约定

远端仓库：4090 `/root/Workspace/xy/DiT`；48 `/home/ds/Workspace/DiT`。计划均在`assets/style_matrix_20261027/matrix64/{4090,48}/`，其中`execution/status.json`、`execution/runs/<arm>/train.jsonl`、`checkpoints/`与`evaluations/`为当前事实来源。两机compile、BF16、有效batch512，禁止梯度检查点。

48旧归档、损坏P1残片及普通版QwenImage21基础模型清理后可用磁盘约167G（df快照），串行每任务启动前有25 GiB空闲空间门槛；之后checkpoint和评测仍会增长，需持续核对。4090仓库盘约1.5T可用。禁止自动删除旧checkpoint。48整盘及各工作区归属见[Disk.md](Disk.md)；详细配置、LR restart、数据配对和历史事故见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)。

## 48关键结果本地备份

白名单11项已于10-28 12:42全部完成，20,020,117,802 bytes逐项尺寸核验通过并生成`completed.json`。本地G盘剩余约65.2GiB。该备份覆盖metadata、DeepSym4 50k/75k/85k、U100 Base55、J0–J3@5k及Qwen A/C 90k；S组新50k与J组未来12k仍是后续追加项。
## 2026-10-29 15:46 暂停扩展并汇总

按用户要求暂停H0/H1实现与启动；双机controller和later观察器SIGSTOP，当前S11训练、S9评测子任务保留，之后不再自动接下一实验。4090 S9已完成50k：val100 SSIM0.59814/LPIPS0.34543/Ink-IoU0.27672，未超过S0；48 J4–J7各5k高度重合，Decoder晋级为空，S11约44.3k/50k，纯训练剩约1.2h。共享训练面板方向cosine约0.886不是留出风格泛化证据；187字差分n_cells=0。完整数据、实际读图、边界及暂停PID见[10-29汇总](docs/1029/RESULTS_AND_PLAN_REVIEW.md)。旧自动队列ETA暂不适用，恢复前重新定计划。

## 2026-10-29 16:50–17:03 新评价实施与收口

S11完成50k及原协议/新指标；其余S0–S9、J0–J7与P1/P2/Deep75/Deep85/U10055均完成首轮重评。合计24个权重终点、30条原生/统一面板记录，IQA三网、E-G跨字检索/MMD、pooled Clean-FID/KID已收口；E-L仅4090子集有权重。报告见[新评价专题](docs/1029/eval_unpaired/REPORT.md)。不是统一187×3seed或6000图终评，OCR/人工校准仍缺失。

S11 val100：SSIM0.59396、LPIPS0.35151、E-G K64 Macro51.25%；187字Macro48.09%。S5−S0检索+7.75pp，探索性95%CI[+0.50,+15.50]；J4–J7无稳定新增收益。通用IQA校准失败，不据此选审美冠军。

两机评测任务结束。4090旧controller已退出，48旧controller及两端later观察器仍暂停；未启动S9FM/Pair、S10、H0/H1，不自动恢复训练。
## 2026-10-29 书写领域指标首轮

已完成缓存特征上的Style-FID/KID、Writer Verification池化AUC/EER、书家宏平均检索mAP，以及k=5 Density/Coverage；两端complete均存在。按用户要求先交付已算指标，暂停扩展实现。官方HWD、CHAED/TabNet尚无分数，不混入结果表。[已完成指标报告](docs/1029/domain_metrics/REPORT.md)，原始统计在同目录及远端`assets/eval_domain_20261029/`。S5的E-G身份核验点估计最好；S7池化Style-FID更低但身份核验不是最好；J2分布有小幅改善，J4–J7仍接近。未进行新生成器训练、未清理权重，无GB级跨机传输。
## 2026-10-29 18:17 S0/S4/S5等曝光85k续训已启动

新有效根：4090 `assets/style_focus_85k_20261029/4090/`串行S0→S4；48 `assets/style_focus_85k_20261029/48_v3/`运行S5。S0已到50,400、S5到50,200，RNG全恢复，GPU真实更新；短窗口约1.61/1.68step/s。当前起S5约5.8h，4090两组合计约12.5h，另加终点评测/写盘波动。

三组各新增35k更新、17.92M样本曝光；B512/micro512，compile，无梯度检查点；统一200步warmup至3e-5、35k周期终值3e-6。每5k保存，85k原val100+统一187×3seed评测。[计划与核验](docs/1029/style_focus_85k/README.md)。旧generation2/later链不恢复，无S5++或Decoder训练。

## 2026-10-31 A/B/C 后训练实验（4090 串行，进行中）

- 启动：~当晚，serial `assets/abc_20261031/plan.json`（--resident-pid 653463）。
- 实测：A 段 1.46s/step（batch 512, compile）。ETA：A@45k ≈ +10.1h；B@75k ≈ +10.1h；C1+C2 ≈ +10.1h；评测 12 面板 ≈ +2.5h。总计 ≈ 34h。
- 评测点：A 25k/35k/45k（用户口径 80k/90k/100k），B/C2 @75000。协议：187×3 seeds + val100。
- 参照（不重训）：S0@85k 内容参照、S5@85k 风格参照。
