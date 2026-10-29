# 21:30阶段指标原始副本

S0–S7为50k/val100，J0–J3为5k/固定187张。各臂保留四个文件：

- `result.json`：原统一评测均值。
- `batch.csv`：逐样本表，保留sample_id，可复算配对差值。
- `geometry.json`：CPU从同批生成/GT图计算的几何统计及逐样本值。
- `writer.json`：独立、source-balanced Writer原型评测。

来源为仓库现存`ANALYSIS/style_repr_injection_matrix_20261026/full_eval_20261028/raw/`，按字节原样复制；文件SHA见上级manifest。没有重新采样或重跑评测。

远端相对仓库路径：

```text
assets/style_matrix_20261027/matrix64/{4090,48}/execution/runs/<arm>/evaluations/<step>/<protocol>/
```

4090臂为S0/S1/S4/S6；48臂为S2/S3/S5/S7和J0–J3。S8在截止时间后完成，不纳入这份原始数据快照。

旧J组外部J2对应内部目标J3，外部J3对应内部目标J4；第二代使用G2J4–G2J7内部编号区分，不能混用。

本副本不含完整生成PNG、checkpoint或逐run代码快照；两张代表性海报位于上级posters目录。逐run训练身份仍应查远端`.done`、`identity.json`、`complete.json`及封存源码清单。
