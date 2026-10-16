# 近期训练代码维护说明

## 活动入口与职责

| 要修改的内容 | 位置 | 约束 |
|---|---|---|
| v88计划 | `tools/analysis/prepare_v88.py::build_config` | 纯函数，import不写文件，CLI才保存配置和anchor |
| 训练与评测构造 | `src/model/config.py`、`src/model/build.py` | 新架构参数必须经同一映射，不能评测端另写一套 |
| Writer×Glyph | `src/model/writer_glyph.py` | 不修改结构token-add，保持checkpoint键名 |
| 局部FiLM | `src/model/components.py` | 老的latent解码调用可缓存；v88不保留无用的带图张量 |
| 后训练目标公式 | `src/loss/writer_objectives.py` | 只负责mask、margin、hinge与CF，无文件I/O |
| 梯度比例与弱监督报警 | `src/train/gradient_budget.py` | 不写参数`.grad`，保持单次clip/optimizer step |
| 后训练输入与采样 | `src/train/strong_style.py` | full-batch、干净标签、共享反事实噪声和t；默认off |
| 书家表优化组 | `src/train/writer_table_tuning.py` | 小LR、WD0，恢复null行；不改变表接口 |
| 分支日志 | `src/train/diagnostics.py` | 只读已有参数与梯度，不重做前向/反向 |
| 同步点汇总 | `tools/v85_compare.py` | 精确匹配step与set，子进程失败/陈旧报告/缺少集合均失败 |

未引入通用插件框架或新的训练调度层。`train.py`保留流程编排，公式、标定、日志分开修改；历史`from src.train.strong_style import writer_margin/norm`导入仍兼容。未来启动脚本读取实际配置的max_steps，不再硬编码180k，也不再重新生成并覆盖已审阅的配置；日志与GPU检查产物按启动时间独立保存。

## 修改和验证

```powershell
python -m pytest -q
python -m ruff check src/train/gradient_budget.py src/train/diagnostics.py src/train/strong_style.py src/loss/writer_objectives.py tools/analysis/prepare_v88.py tools/v85_compare.py tests/test_experiment_maintenance.py tests/test_writer_glyph.py tests/test_setup_runtime.py
```

pytest默认只收集`tests/`。GPU脚本和第三方基线命令不是单元测试，按各自入口手动运行。CPU开发机上的运行时目录测试仅替身CUDA设备选择，不把正式训练改成CPU模式；Windows原生日志的非UTF8字节以转义形式保留。

准备新默认计划时显式运行`python tools/analysis/prepare_v88.py`，会生成配置与预训练表anchor；修改并审核JSON后再用`bash tools/analysis/run_v88_base.sh [配置路径]`启动。不要为了缺失anchor直接覆盖一份已经调过的计划。单独GPU检查支持`python tools/analysis/smoke_v88.py --config 配置路径 --out 独立输出目录`。

新增目标先测公式与梯度，再接训练hook；新增模型路径先测零门控等价、梯度可达、CFG丢弃和strict加载，再做GPU整批检查。不要以“范数非零”直接宣布结构有效。无重计算、无microbatch；正式显存检查使用与训练相同的compile配置。

内容指标支持`--step-map name=5000`；规范样本目录`step0005000`可自动提供step。CSV通配符必须只命中一个文件，缺少set或存在多个步点时不能回退到最后一行。独立Writer的CPU/GPU口径以及LPIPS-VGG/Alex分开记录。

## 本轮审查与重构记录

范围为近期v86后训练、v88及其评测工具，采用按职责拆分、纯配置构建、兼容重导出和可恢复归档；不大改训练主循环或旧模型定义。

| 问题 | 修复前 | 修复后 | 风险 |
|---|---|---|---|
| 评测失败被误当成功 | 子进程失败仍读取旧JSON | 校验退出码、报告更新时间和完整集合 | 高，结果可信性 |
| 指标步点错配 | CSV默认最后一行，缺set可回退 | 显式step/set唯一连接，GPU评测前预检 | 高，结果归因 |
| 配置import写文件 | `prepare_v88`顶层写配置 | `build_config(base)`返回新字典，CLI负责I/O | 中，隐蔽副作用 |
| 公式与训练状态混杂 | `strong_style.py`同时负责所有计算 | 纯目标、梯度预算、训练hook各自独立 | 中，可维护性 |
| 关闭模式歧义 | 空字符串可误启后训练 | 工厂只接受off/A/B/C/D，off不加载Critic | 中，启动安全 |
| 无用图缓存 | 空间FiLM为latent解码缓存v88不使用的张量 | v88关闭缓存，老接口默认保留；输出逐位一致 | 低，checkpoint键不变 |
| 测试发现范围过宽 | 导入第三方CLI导致SystemExit | 默认tests目录，未屏蔽项目测试 | 中，验证可用性 |

这轮改动保持目标公式、梯度累加顺序与checkpoint键名；不声称已验证重构后GPU吞吐。远端v88未热更新。旧模型与大模块仍有技术债，但不能仅因当前配置未启用就删除。

## 验证结果

- **2026年10月15日阶段快照**：本地 `python -m pytest -q`：**162 passed，47.64秒**。
- **2026年10月16日最新门禁**：本地全套测试 **464 passed**（41.75秒）。
  - 前向黄金校验 `tools/verify_forward_parity.py` exit 0 ✓
  - 梯度黄金校验 `tools/verify_grad_parity.py` **59,818,645 个梯度元素逐位一致** ✓
  - 训练冒烟 `tools/smoke_train.py`（前向+反传+优化器两步）通畅 ✓
  - 预检 `tools/preflight.py --strict` 0 失败，`tools/lint_sync_hotpath.py` 0 违规，`ruff` 0 错误，`pyflakes` 0 未定义名。
  - 评测真机实测提速 2.4×（178s -> 73s，RTX 4090），C2OT 彻底清理，六项评测正确性不变量落地。

保护测试覆盖逐位FiLM等价和checkpoint键、零门控主干等价、P/Q梯度、CFG掩码、完整batch反事实输入、顺序反传等价、梯度标定公式与弱监督报警、36文件哈希、配置import无写入、错误模式关闭、失败/陈旧/不完整评测报告和step/set唯一连接。GPU路径未重跑，避免占用正在训练的4090。

归档源代码38个文件共131,821字节，其中36个原始文件加README和manifest；历史原生CSV、独立评分JSON、三张退步海报在实验文档旁保存。既有未提交改动未重置，也没有将它们混入新提交。
