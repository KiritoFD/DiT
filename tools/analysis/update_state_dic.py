with open('state.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update queue section to mark G_writer_only cancelled
text = text.replace(
    'Base完成后，G_writer_only从同30k恢复至100k',
    '已取消串行任务 G_writer_only；Base 训练至 100k 后自然收尾'
)
text = text.replace(
    '两组任务仅 `baseline_resume → G_writer_only`。第二组用`--resume-expanded`，按参数名保留旧Adam slots及旧EMA；新参数无历史slots。当前Base的EMA未重置，完整配对214/214。',
    '原队列计划 `baseline_resume → G_writer_only` 已调整：依据用户指令，已于 05:32 移除 `G_writer_only` 并创建日志哨兵拦截，Base（PID 3162410）独立训练至 100k。'
)
text = text.replace(
    'Writer-only尚未启动，不能判断调制收益',
    'Writer-only任务已依据用户指令取消（已清理队列计划），不再执行'
)

# Update implemented section
old_impl = '''## 5. 已实现、但不能当成已完成实验

- Writer-only AdaGN：已实现，待G组训练；仅新分支从头初始化。
- 条件残差gate：`residual_cond_gate=false`默认关闭，初始增益1；仅实现，不入当前队列。
- Skip/STD门控、Decoder[3,3,2]、全网3块：实现保留，当前不跑。'''

new_impl = '''## 5. 已实现、但不能当成已完成实验

- **DiC 条件残差门控 (`residual_cond_gate`)**：已实现并在本地与 Node 48 通过双端回归测试。基于 $y = \\text{Skip}(x) + g(c) \\odot \\text{Conv}_2(h)$，采用 bounded gain（$[0.5, 1.5]$）与 zero-linear 初始化，步 0 输出与基线完全逐位一致（Max diff = 0.00000000e+00），参数量 75.04M（+2.95M, +4.1%）。
- **DiC 分尺度独立条件 (`stage_specific_cond`)**：已实现并在双端通过回归测试。针对 32×32、16×16、8×8 三级分辨率独立分配条件表示，全局同步 Writer/Script dropout 掩码杜绝跨尺度条件泄露。支持两类模式：
  1. `stage_cond_mode='adapter'`：零初始化线性残差适配器 $c_i = c + \\Delta_i(c)$，步 0 与基线输出绝对恒等（Max diff = 0.00000000e+00），参数量 72.88M（+0.79M, +1.1%），完美兼容现有 25k/30k/50k checkpoint 热启。
  2. `stage_cond_mode='independent'`：DiC 原生每尺度独立 projection 模块（time/writer/script），参数量 72.80M（+0.71M）。
- **DiC 激活函数切换 (`act_fn`)**：支持 `'silu'` 与 `'gelu'` 一键切换，默认保持 `'silu'` 保证严格兼容。
- **预置实验配置**：已在 `src/unet/configs/connectivity/` 固化生成 `DiC_residual_gate.json`、`DiC_stage_cond_adapter.json`、`DiC_stage_cond_independent.json`、`DiC_combined.json` 4 份标准配置。
- **取消项**：原计划串行的 `G_writer_only` 已从队列安全剥离。'''

text = text.replace(old_impl, new_impl)

with open('state.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('state.md updated with DiC switches and cancellation.')
