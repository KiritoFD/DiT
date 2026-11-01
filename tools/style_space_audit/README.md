# 同一Style Disc空间审计

只做判别、分布与冻结梯度诊断，不启动生成器训练。输入必须是已核验的语义Writer身份和精确配对样本；拒绝按0..9猜旧checkpoint的书家顺序。

## 两机步骤

4090导出已有P2 CFG1/1.6、GT、swap及确切STD/noise，不重新采P2：

```bash
python -m tools.style_space_audit.data --root assets/style_space_audit_20261026 --export
```

将该目录的小体积产物传48，使用本机原生Deep114@85k与`assets/style_disc_ce_holdout.pt`：

```bash
python -u -m tools.style_space_audit.run --root assets/style_space_audit_20261026
```

运行器依次进行来源/像素去重审计、Deep共享STD/原生STD、全模型特征、固定GT-PCA、原型检索/分布/方向、严格留出VAE重建及20字梯度pilot。Deep/梯度已有产物可跳过；冻结特征缓存按数量校验复用，若改变模型或预处理请用新输出目录。

## 契约

- 原编码器为26槽位CE ResNet18，使用其源实现严格加载；不以Pixel D1冒充它。
- 精确复现原楷五家/行四家ARI，并揭示原Qwen/DINO常数；保存原脚本而不覆盖。
- 聚类在256D归一化空间，PCA只由GT拟合。Qwen/P2/Deep使用同100个目标；Qwen按Writer/Script/字符/像素SHA256连接。
- 来源为权重保存的目录proxy，非已验证碑帖ID；原训练CSV哈希缺失，明确证据限制。
- 梯度pilot使用可微Torch bicubic160；冻结图像评价和VAE对照使用原PNG/PIL160管线，两者不混称逐位相同。
- 独立评价器和该ResNet共享部分数据，网络不同不等于统计独立；pilot单seed不充当最终训练验收。
- 白底/墨量控制改变图像，墨量匹配还改变几何；不将分数变化作纯因果分解。
- 程序不负责自动恢复此前暂停的训练。

报告与图：[ANALYSIS/style_space_audit_20261026](../../ANALYSIS/style_space_audit_20261026/README.md)。输出JSON含逐图/逐槽位指标，避免仅有漂亮散点而没有身份和覆盖信息。
