# P2收尾对照

`run.py`复用共享U-Net训练器，按A strong→B mild执行两个独立5k阶段；B仍从原P2 EMA起步。

```bash
python -m tools.p2_consolidate.run --checkpoint <P2.pt> --out <new-output>
```

默认114M、B512、LR5e-6/条件2e-5、pair .3、seed90、2.5k/5k存评。新输出必须不存在status，不隐式覆盖重跑。
`std_shape_step_offset`延续原阶段的增强配比时钟；原模型严格加载全部权重，只有训练增强模式允许strong↔mild变更。

`direction.py`仅CPU读取已有swap PNG及同字同书体GT，计算墨量/笔宽/结体相对方向，明确记录缺失覆盖及GT来源。
`--extra`不带路径可只用训练GT复核；不会训练分类器或触碰生成模型。

`evaluate.py`在两组完成后复用内存评测落盘PNG，不重新采样，输出完整100图海报和预先固定的正常/问题示例。
不以不可靠的Writer probe判定书风，不自动切换默认模型、不自动蒸馏。

设计及本轮数据见`ANALYSIS/p2_std_consolidation_20261025.md`。
