# P0 → S4 部分迁移：能力已实现并逐张量验证

**状态：** 迁移能力**已完成并验证**；训练**未启动**。
**代码：** `tools/p0_s4_ft/{partial_transfer.py, migrate_dryrun.py}`
**测试：** `tests/p0_s4_ft/test_partial_transfer.py`（9 项全通过）
**证据：** `assets/p0_s4_ft/migration_plan.json`

## 一、方案前提的两处修正（实测，非推测）

方案假定 P0 是 **R2/I3**「预计为 R2/I3，但历史报告存在配置冲突，必须先从实际 checkpoint 确认」。
已从实际 checkpoint 确认，并发现第二处冲突：

| | 方案假定 | **实测** |
|---|---|---|
| P0 注入 | I1（表格写「I1 Spatial AdaGN 新建」）| **I3**（`style_matrix_injection=3`）|
| P0 repr | R2 | **R2** ✓（`style_matrix_repr=2`）|
| P0 writers | 47 | 47 槽位 → 表 48 行（含 null）|
| S4 repr/inj | R2/I1 | **R2/I1** ✓，`num_writers=10` → 表 11 行 |

**关键推论：I1 与 I3 的参数名集完全不相交。**

```
I1 (kind==1):  style_injections.N.local.0.{weight,bias}   Conv2d(C+std+64, 64, 1)
               style_injections.N.local.1.{weight,bias}   Conv2d(64, C*2, 3, p=1)
I3 (kind==3):  style_injections.N.down.weight   basis.weight   output.weight
               style_injections.N.mix.{weight,bias}
交集（除 gain buffer）= 空
```

所以从 P0 切到 S4 **不是「保留 R2、只新建 I1」**，而是**「保留 R2、丢弃全部 I3」**。
这条已向用户确认，结论：**接受丢弃 I3，I1 全新初始化**。

## 二、为什么必须新增迁移模式

现有 `transfer_style=True` 会加载**全部** `style_representation.*` / `style_injections.*`，
且对任何残留键**抛错**（`common.py:42-45`）。用于本次迁移会在 18 个 I1 张量缺失上直接失败。
设为 `False` 则连 R2 一起丢掉。两者都不可用 —— 这正是方案判断正确的地方。

## 三、迁移契约（可审计，不是「看起来对」）

静默失败是这类迁移的真正风险（前缀打错、模块改名、repr/injection 不匹配）。
所以契约是**可逆的、强制的**：

1. 每个源风格张量**要么 TRANSFER 要么 DROP**，逐名列出；
2. 每个目标风格张量**要么 TRANSFER 要么 FRESH**，逐名列出；
3. **不在任何清单里的张量直接抛错** —— 没有默认行为；
4. 形状不匹配抛错，不静默跳过；
5. 迁移后**逐张量字节比对**，不只依赖 `load_state_dict` 的返回值。

DROP 只允许用于已知的架构差异，不是兜底。

## 四、实测结果（P0 50k EMA → S4 架构）

```
totals: transferred=18  dropped=16  fresh=12

backbone:  loaded 333 tensors, byte-identical 333/333 (max_diff=0.000e+00)
style:     transferred 18 tensors verified byte-identical (max_diff=0.000e+00)
writer:    backbone.writer.weight              48 -> 11 rows
           style_representation.table.weight   48 -> 11 rows
OK: backbone + R2 byte-identical to P0; only writer rows and I1 differ
```

- **333/333 主干张量字节相同** → P0 的通用内容先验完整保留（方案的核心前提成立）
- **18/18 R2 张量字节相同** → 已训练的 LocalCNN 继承
- **16 DROP** = I3 的 down/basis/output/mix（I1 中结构上不存在）
- **12 FRESH** = I1 的 local.0/local.2，全新初始化

## 五、迁移中发现并修掉的两个真实问题

**(1) 有两个 writer 表，不是一个。** 方案只提到 Writer Embedding 一处，实际存在：

```
backbone.writer.weight              U-Net 读取的 writer 表
style_representation.table.weight   R2 投影用的 writer 表
```

**两个都随 `num_writers` 变**（48→11 行）。只处理一个会在主干拷贝时留下形状不匹配的静默隐患。
现两者都由 `init_writer_tables` 统一从源行**均值 + 小扰动**播种，并**单独报告**，
不计入风格张量的 transferred/dropped 计数（否则审计表会骗人）。

**(2) `init_writer_table` 的行数假设错误。** 我最初按 `num_writers` 断言行数，
但表宽是 `num_writers + 1`（null 槽）。已改为按实际张量形状取行数。

## 六、尚未做的事（诚实记账）

- **训练未启动。** 本轮只完成了「训练前必须通过的检查」#1（权重迁移）。
- **检查 #2–#4 未做**：null Writer 下输出应与 P0 基本一致（需前向）、
  Writer 有效性（需 100–500 步训练）、训练契约落盘。
- **`common.construct` 尚未接入本迁移**。当前 `partial_transfer` 是独立模块 + dry-run；
  要让真实训练用上，需要在 `construct` 里加一个分支（例如
  `spec['initialization']=='partial'`）调用 `load_backbone` + `apply_transfer` +
  `init_writer_tables`。**这是下一步的必做项**，也是唯一还需改动的生产路径。
- **未执行 gain 校准**（方案要求：新 I1 暂不自动校准，避免放大随机初始化扰动）。符合方案。
- **方案剩余部分**（S4-W/S4-FM/S4-Tail 三阶段、J2/J3 分叉）均未实现。

## 七、复现

```bash
PYTHONPATH=. python tools/p0_s4_ft/migrate_dryrun.py --device cpu
# 4090 CPU 即可，约 1–2 分钟；输出 assets/p0_s4_ft/migration_plan.json

python -m pytest tests/p0_s4_ft/ -q     # 9 项，1.7 秒
```

单测覆盖的是**判定逻辑**（用小型替身模块）：逐张量分类无遗漏、
repr/injection 不匹配必须抛错而不是猜、**无法解释的源张量必须拒绝丢弃**、
迁移与主干拷贝逐字节精确、writer 表单独报告且行数正确。
真实 P0 EMA 的端到端证明在 `migrate_dryrun.py`（上文第四节结果）。
