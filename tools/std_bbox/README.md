# STD bbox归一化修复（2026-10-22）

## Bug 分析报告

### 一、错误现象

旧 `std_soft_final` 采用固定em字号居中，而GT/海报在256画幅展示；STD墨迹没有填满可用画幅。
复现：读取index对应PNG，计算非零墨迹bbox最长边/64。8945个字的最小/中位/最大为0.359375/0.703125/0.8125。
这不是只调整海报的显示比例就能修复的问题，模型64条件本身也带着这些多余空白。

### 二、根因分析

自顶向下定位到 `src/utils/std_soft_glyph.py::render_base_mask`：旧实现明确禁止bbox缩放，只按font_ratio=0.79渲染。
修复采用实际非零墨迹bbox，而非字体textbbox空白；等比例最长边填满256，短边居中补白，无额外边距。
不是横纵分别拉伸；“一”等扁字保持原始字形比例。随后BOX下采样得到64条件，再按需要模糊。

### 三、最小化Diff格式修复补丁

```diff
+ fit_ink_bbox(image, size=256)
+ canonical = fit_ink_bbox(complete_glyph_render, 256)
+ condition = canonical.resize((64,64), BOX)
```

通用字体生成器默认采用新bbox路径；`--legacy-em`显式复现旧em规则。
批量修复从旧index的精确font_path/font_face重新渲染，不重新选择字体，也不从旧64图放大生成伪高分辨率图。

```bash
python -m tools.std_bbox.build --source assets/data_cohorts/top20_20261021_v1/std_soft_final --out assets/data_cohorts/top20_20261021_v1/std_bbox_v1
```

4090已完成全部8945字：`std_bbox_v1/display256`为原始256缓存，`std_bbox_v1/condition64`为模型条件缓存。
两个目录各含index/provenance，根目录glyph_audit.csv逐字记录新旧bbox、字体；audit.json含字体和源index哈希。
全量核验最长墨迹跨度分别为256、64。旧缓存、历史checkpoint配置和海报不覆盖，避免改写旧实验身份。

### 四、防回归测试用例

`python -m unittest discover -s tests -p test_std_bbox.py -v`：3项通过。
覆盖矩形bbox填满/居中、极扁笔画保留比例、拒绝空白图；不做模型训练测试。
对照图已实际检查：本地 `assets/audits/std_bbox_20261022/preview.png`，同目录有全量audit和逐字CSV。

### 五、预防建议

STD缓存明确区分“字体渲染尺寸”与“墨迹bbox几何”；预览必须来自实际模型缓存，不能单独在海报里纠正。
未来推理显式指定新根目录，评测工具已支持：

```bash
python -m tools.cohort_eval.run --checkpoint CHECKPOINT --collection top20/test --out FRESH_OUTPUT --std-root assets/data_cohorts/top20_20261021_v1/std_bbox_v1/condition64 --std-sigma 0
```

本轮只完成预处理与缓存修复：固定模糊新训练在约300步被用户叫停，无5k checkpoint；旧几何无模糊评测也已停止。
未自动开始bbox推理或重训。旧模型没有见过新几何分布，未来测试属于输入分布干预，不能把差异直接当作重训收益。
