# 字形海报与粗细校准

两个独立 CPU 工具，不加载生成模型、不修改训练、不覆盖原图。依赖 NumPy、Pillow、SciPy、scikit-image；下面命令在仓库根目录运行，JSON 中路径也相对运行目录解析。

## 1. 粗细校准

测量：灰度图转墨迹强度 `1-gray/255`，默认阈值 0.5；在确定性 medial axis 上统计 `2 × EDT` 的 P10/P50/P90（单位为原图像素）。同时记录墨量、骨架长度、等效宽度、连通块和孔洞。P90/P10 是**宽度反差，不是提按/笔压**。种子固定为 0，并记录软件版本；跨版本不保证逐位一致。

### 手动控制

```powershell
python -m tools.glyph_review.calibrate apply --csv exp-std/csv/eval_top5_fixed.csv --input assets/results/top5_u100_base_50k --output artifacts/width_manual --mode manual --delta -1 --indices 18,19,20
```

`--delta` 是边界法向位移：正数增粗、负数变细，**不是最终笔宽差**；也可改用 `--p50 10` 指定中位宽度目标。默认 `--max-shift 2 --strength 1`；`--strength 0` 完全不改图，连 PNG 文件字节也保留。

### 训练集参考（可用于实际后处理）

```powershell
python -m tools.glyph_review.calibrate reference --csv TRAIN.csv --image-root . --role train --output artifacts/width_reference.json
python -m tools.glyph_review.calibrate apply --csv EVAL.csv --input GENERATED_DIR --output artifacts/width_reference_output --mode reference --reference artifacts/width_reference.json
```

训练图片必须是与评测一致的 256×256 画布（其他尺寸需显式 `--size`）。按书家×书体拟合**每图三分位数的平均值**，不是全分布匹配；不隐式复用旧表，不用全局平均兜底缺失槽位。`--role train` 是调用方的数据声明，不自动证明划分正确。提供 `--gt-dir` 后会检查参考图与评测 GT 的文件哈希重叠；未提供 GT 时不能认证无泄漏，重编码后的同源图也需要调用方按身份排除。

### GT-oracle（仅诊断，不是无参考生成成绩）

```powershell
python -m tools.glyph_review.calibrate apply --csv exp-std/csv/eval_top5_fixed.csv --input assets/results/top5_u100_base_50k --output artifacts/glyph_review/top5_oracle --mode oracle --gt-dir artifacts/DiC_U100_review_20261020/images/reference/gt --gt-csv exp-std/csv/eval200_fixed.csv --indices 18,19,20,22,25,26,27,28,29,34,35,36,39,40,41,45,46,47,48,50
```

不同 CSV 用规范化 `image_path` 对齐，不按行号硬配。`oracle` 逐字取 GT 目标；海报自动标明“GT-ORACLE 仅诊断”。

默认 `--method uniform` 搜索统一位移；`--method profile` 按三分位数插值局部宽度调整，仅支持完整三分位数目标。两者都是有界的原图法向灰度/RGB变形，**不把二值 mask 重画成灰度图**。远离边界的纹理不动，边界附近会重采样；不保证任意目标可达。

安全规则：对保存后的 uint8 图重新测量，检查 8 连通墨迹、4 连通背景孔洞的一对一对应，避免只数数量而漏掉同时分裂/合并。默认 `--min-area 1` 检查全部；调高此值会显式忽略小结构。目标模式只接受宽度误差降低的安全候选，否则原样回退。少于 25 个墨迹像素的图不校准。**拓扑相同不等于内容正确**，仍需看图，不能承诺绝无局部变形。

产物：`gN.png`、`metrics.csv`、`records.json`、`run.json`。记录目标、前后指标、有效位移、回退原因、文件哈希及协议。`run.json` 只有全部成功才标记 complete。提供 GT 会补灰度 SSIM、Ink-IoU；`--lpips alex --device cuda` 可选增加 LPIPS（需安装 torch/lpips，可能首次下载权重；失败不会填假零）。工具不会选择最优 checkpoint 或自动宣布综合质量提高。

`run.json` 的 summary 在有 GT 时汇总 P50 平均绝对相对误差、SSIM、Ink-IoU（及选用的 LPIPS），只覆盖本次选定样本。默认 2px 最大边界位移不可能把 25px 笔画直接变成 10px；需要更强调整时显式增加 `--max-shift`，但安全检查可能因此拒绝更多候选。

## 2. 海报

```powershell
python -m tools.glyph_review.poster --config tools/glyph_review/examples/raw.json --output artifacts/glyph_review/raw_poster
python -m tools.glyph_review.poster --config tools/glyph_review/examples/calibrated.json --output artifacts/glyph_review/calibrated_poster
```

每格完整画布以相同缩放显示，不裁切、不重心对齐、不隐式校准。STD 缩略图只作条件展示，可显式统一放大，不测它的原图笔宽。生成图尺寸不对直接报错。默认 8 行分页，可用 `--rows-per-page 20 --cell 256` 输出单张大图。

配置包含 anchor CSV、列目录/各自 CSV、明确的行选择。`selection` 支持 `indices`、`policy: all` 或显式 `intersection`。模型列必须填写 architecture、step、state（EMA/RAW/UNKNOWN）和 cfg，不从文件名猜测；未知采样身份需标 UNKNOWN 并先核对再做严格性能比较。

每格显示 P50/P10/P90、墨量、连通块、孔洞，校准列增加状态。`manifest.json` 保存样本身份、覆盖范围、源文件路径/哈希、测量和校准记录。给出的 20 字示例是历史海报选字，**不是全部 187 字评估**。字体优先仓库 SimHei、Windows 微软雅黑；也可 `--font C:/path/font.ttf`。

示例区分 `Top10-A114M`（Writer 解耦实验）和 `Deep114M`（普通深网），不能混称。示例 state/CFG 中 UNKNOWN/null 表示源目录本身不足以认证身份，不把它猜成 EMA/1.0。

本次已生成的示例：

- [原始对照第1页](../../../artifacts/glyph_review/raw_poster_final/page01.png)（共3页）。
- [校准前后第1页](../../../artifacts/glyph_review/calibrated_poster_final/page01.png)（共3页，GT-oracle）。
- 原始结果索引分别为上述目录中的 `manifest.json`；校准逐样本数据在 `artifacts/glyph_review/top5_oracle/metrics.csv`、`top10a_oracle/metrics.csv`。

默认严格保拓扑、最大位移2px的20字演示中，Top5有7张、Top10-A有4张被调整，其余原样回退。只是工具安全行为演示，不能据此宣布粗细问题已解决。

## 最小验证

```powershell
python -m pytest tests/test_glyph_review.py -q
```

只测确定性、零强度恒等、增粗/变细、拓扑回退、参考泄漏和海报样本对齐。旧海报/校准脚本保留供历史复现；新工作从这两个入口执行。
