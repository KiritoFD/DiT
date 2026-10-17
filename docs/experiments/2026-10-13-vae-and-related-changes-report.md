# 从 SkelNet 到字符表与 VAE 的阶段回顾

时间范围：10.1—10.13。

**主线：skel 没有成为有效的中间接口。两阶段、latent 骨架监督和 12ch 联合目标都没带来可靠收益，因此转向对比学习字符表；V68 达到约 0.61 后，再把重点移到 VAE 细节保真与推理采样。** 这里“skel 不能用”指本阶段试过的这些路线，不是否定所有骨架先验。

## 第一阶段 两阶段效果不好 中间 latent skel 很难监督

10.1 左右，尝试“标准骨架→SkelNet 预测书家骨架→stage2 渲染”。问题卡在中间目标：latent loss 看着收敛，解码后的细线、位置和连通性却不可靠。V45 的骨架 SSIM 超过直接用标准字的基线，IoU 反而更低。增加容量、换注入、补结构 loss，都没有形成稳定的下游收益；10.3 封存两阶段主线。

![两阶段 SkelNet 阶段汇总](phase_review_20261013/01_skelnet.png)

下面补上 SkelNet 时期留下的实际推理拼图。输入和目标仍能看出字形，预测列却出现大面积碎片化；这比单看 latent loss 更直接地说明中间骨架不适合作为可靠接口。原图未嵌入具体 run／step，因此只作定性证据；v38／v44／v45 的定量结果另列在海报下部。

![SkelNet 实际推理结果与各 run 指标](phase_review_20261013/01_skelnet_runs.png)

## 第二阶段 Moyi 复现 12ch 不如简单的 4ch

复现 Moyi 的 image + Canny + skeleton 12ch 联合扩散，并做纯图像 4ch 对照。同为 50k，4ch 的 SSIM / LPIPS 略好；继续到 80k，SSIM 达 0.6085。12ch 并非所有指标都差，但没有兑现额外结构监督的收益。**继续把 skel 加进生成目标，并没有让主线更好。**

![Moyi 12ch 与 4ch 阶段汇总](phase_review_20261013/02_moyi.png)

## 第三阶段 用对比学习字表替代 std skel 条件

参考三因素条件设计，保留书家、书体，用对比学习字符表提供字形表征，不再要求模型先生成或严格遵循一条骨架。三表与主干条件路由形成 V68 主线，SSIM 达到约 0.61。后续 std-skel 回测也未超过字表路线；“学字形表征、直接生成图像”比中间骨架接口更有效。

![字符表路线与 V68 阶段汇总](phase_review_20261013/03_character_table.png)

海报是不同进度的路线展示，不是单变量涨幅；同 65k 的记录为 V68 0.5833、V70 std-skel 0.5535。

## 第四阶段 达到 0.61 后 怀疑 VAE 的细节上限

VAE 的演进不是一次完成的，而是三步：

1. **第一次直接把 DINO 接到 VAE。** 配方是 `L1 + KL + DINO_struct + DINO_style`，同时去掉了 LPIPS 和判别器。PSNR／SSIM 虽然提高，但顶带高频相对 SD-VAE 从 0.816 降到 0.783，细笔、飞白反而被磨掉。这里能确认的是整套配方丢高频，不能把因果只归给 DINO 单项。
2. **随后补高频监督并校正 latent 合同。** DoG 的早期增益很小；直接优化 top-band 才把高频从 0.783 拉到约 0.828。同时发现旧 shard 沿用了 SD 的 scale，存盘 latent std 只有约 0.20，于是补齐 scale、shift 与解码约定。
3. **最后从干净 SD-VAE 底座重训，回到 LPIPS + PatchGAN。** 删除 DINO，使用 `L1 + LPIPS + Laplacian + topband + KL`，PatchGAN 在 5k 步后启用并用自适应权重。最终同批 N187 往返 PSNR 30.56→34.04 dB、顶带份额 0.783→0.901。

**结论：压缩器的高频上限确实提高了，但重建收益没有等幅变成生成收益；而且本轮 GAN 权重最终很小，PatchGAN 的独立贡献尚未由无 GAN 消融证明。**

![VAE 训练与重建阶段汇总](phase_review_20261013/04_vae.png)

为避免只看 VAE 往返重建，下面把 VAE 更替后几个代表 DiT run 的实际生成结果并排放在一起。它们的训练步数和其他配置并不完全相同，因此用于观察阶段演进，不作单变量归因。

![VAE 相关 DiT run 推理结果墙](phase_review_20261013/04_vae_runs.png)

## 第五阶段 SDE free lunch 观感改善但会丢笔

同期从 10.8 起尝试不重训的 SDE 采样：V68 的输出更干净、平均 SSIM 上升，但部分细笔会断掉或消失。下图后两列直接显示局部缺笔；全批 LPIPS-alex 和 Ink IoU 反而变差。**视觉更好看，不等于字写得更完整。**

![SDE 观感改善与局部缺笔对照](phase_review_20261013/05_sde.png)

![SDE 跨 run 推理结果墙](phase_review_20261013/05_sde_runs.png)

这组现象来自旧 SDE；后续修正漂移、在物理 VAE 的 V80 上复核，没有证明稳定的额外 SDE 感知收益。目前验证到的推理收益主要来自 CFG 调整，不能把 SDE 当所有模型的通用 free lunch。

## 阶段结论

**先放弃强 skel 接口，字符表把主线推进到 0.61；再改善 VAE 与采样，但始终同时看观感、LPIPS 和笔画完整性。** 原图只用于排版，没有重新生成、锐化或修笔。每张海报注明指标来源、实际 step 和比较限制。

[详细数据与工程审计附录](2026-10-13-vae-and-related-changes-technical-appendix.md) · [海报来源与哈希](phase_review_20261013/sources.json)
