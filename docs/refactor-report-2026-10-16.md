# 重构报告：Callig-DiT 训练/模型/评测三大主线的结构收敛

> 依据专家审计《Callig-DiT 全面代码审计与单卡 4090 重构方案》的 PR0→PR7 计划执行。
> 本文档按 `python-refactoring` 技能的《重构文档生成》模板编写，用于**项目归档**与**团队分享**。
> 逐日过程记录见 [refactor-pr0-pr4-progress-2026-10-16.md](refactor-pr0-pr4-progress-2026-10-16.md)。

---

## 1. 概述

| 项目 | 内容 |
|------|------|
| **重构范围** | `src/train/`（训练主函数、CLI、配置）、`src/model/`（DiT 条件系统）、`src/eval/`（评测编排、海报、CLI）、`tools/`（冗余清理） |
| **重构规模** | **大型**（跨模块/包，按审计的 PR0–PR7 分 8 阶段执行） |
| **影响文件数** | **287 个文件**（含 200 个配置归档的重命名、23 个逐字冗余删除） |
| **代码行变化** | **+10,446 / −5,569（净 +4,877 行）**<br>⚠ 净增主要来自**测试与新增模块**：`tests/` +4,825（28 个重构测试）、新增生产模块 ~1,650 行（`trainer.py` 984 / `dit_condition_mixins.py` 343 / `tools/prof/bench_train.py` 等）；**核心单元全部大幅缩小**（见 §5.1） |
| **风险等级** | **低**（全程保持 `state_dict` 键与注册顺序不变，以**逐位黄金**裁决；未改动优化器实现） |

### 关键单元行数变化（实测）

| 单元 | 前 | 后 | 变化 |
|------|----|----|------|
| `src/train/train.py:main` | 2,429 | **1,549** | **−880** |
| `src/train/train.py` | 2,742 | **1,995** | −747 |
| `src/train/trainer.py` | — | 984 | **新增** |
| `src/model/dit.py:__init__` | 870 | **481** | **−389** |
| `src/model/dit.py` | 1,531 | **1,314** | −217 |
| `src/model/dit_condition_mixins.py` | — | 343 | **新增** |
| `src/eval/in_mem_eval.py:run_in_mem_eval` | 388 | **106** | **−282** |
| `src/eval/in_mem_eval.py` | 1,487 | **1,139** | −348 |
| `src/eval/cli.py:build_parser` | 866 | **32** | **−834** |
| `src/train/cli.py:build_parser` | 1,193 | **77** | **−1,116** |
| `src/train/configs/*.json` | 304 | **104** | −200（归档，历史保留） |

---

## 2. 重构动机

审计给出的核心判断：**首要问题不是模型计算量，而是研究实验分支、生产训练路径、历史兼容逻辑与评测基础设施高度耦合**——已建立若干拆分，但**未建立稳定的模块边界**。

具体后果：

1. **每次增加实验条件都要改多处**，且**难以证明没有引入语义变化**。
2. 三个文件把多种职责压在一个函数/类里：`train.main()`（2,403 行）、`DiT_2Cond.__init__`（870 行）、`in_mem_eval`（生成/解码/指标/海报/缓存/CSV 混在一起）。
3. 审计点名的 **11 项正确性缺陷**（P0-1 评测身份、P0-2 `np.empty` 未初始化、P0-3 STD 静默映射第 0 行、P1-1 前向中创建参数、P1-2 RNG 未完整保存…）会让**实验结果失真**——在继续投入 v88 等新基模训练前必须先修。
4. 冗余持续累积：**304 个 JSON 配置**（活跃仅 14）、`tools/` 内大量重复。

**不重构的后果**：技术债继续累积，实验身份不可强制验证（此前已发生 V80 图像被当作 v84A 基线的事故），微增益（5k–10k 步）的比较失去可信度。

---

## 3. 问题清单

问题类型使用 `code-smells.md` 的标准分类；严重程度按审计的 P0/P1 与本项目的实际风险判定。

| # | 问题类型 | 位置 | 严重程度 | 说明 |
|---|----------|------|----------|------|
| 1 | 过长函数（Long Function） | `src/train/train.py:main`（2,429 行） | 高 | 模型/数据/DINO/Critic/正则/EMA/编译/梯度探针/恢复/保存/评测全在一处 |
| 2 | 过长函数 | `src/model/dit.py:DiT_2Cond.__init__`（870 行） | 高 | block 流、书家/书体/字符表、STD 编码、条件路由融合、空间风格、Skeleton/Deform 遗留、CFG、checkpoint 兼容全在一处 |
| 3 | 过长函数 | `src/eval/in_mem_eval.py:run_in_mem_eval`（388 行） | 高 | 采样 + 解码 + 指标 + 制品 + CSV 编排混在一起 |
| 4 | 过长函数 | `src/eval/cli.py:build_parser`（866 行） | 中 | 253 条 `add_argument` 平铺 |
| 5 | 过长函数 | `src/train/cli.py:build_parser`（1,193 行） | 中 | 同上，规模更大 |
| 6 | 过长参数列表 | `EvalSetContext` 缺位处 | 中 | 单 set 评测需 22+ 个游离局部 |
| 7 | dict/tuple 传递复合数据 | 训练主函数 | 中 | 单步训练状态以 60+ 个游离局部流转，无显式所有者 |
| 8 | 过大模块 | `src/eval/in_mem_eval.py`（1,487 行） | 中 | 海报生成与评测编排同文件 |
| 9 | 重复代码（跨文件） | `tools/legacy/*`、`tools/evaluation/*` | 中 | 23 个**逐字相同**副本 |
| 10 | 重复/历史配置 | `src/train/configs/*.json`（304 个） | 中 | 活跃仅 14 个 |
| 11 | 正确性缺陷（P0/P1） | 审计 §2 的 11 项 | **高** | 会使实验结果失真 |
| 12 | 上帝类 | `DiT_2Cond`（1,347 行） | 中 | 构造与条件系统未分离 |

---

## 4. 重构策略

### 采用的模式

| 问题 | 应用的重构模式 | 参考 |
|------|----------------|------|
| 过长函数（1–5） | **Extract Method**（按**连续区段**抽取，保持调用顺序） | `code-smells.md §1.1` |
| 过长参数列表（6） | **dataclass 参数对象**（`EvalSetContext` / `TrainState`） | `design-patterns.md §1.3` |
| 复合数据游离（7） | **dataclass 类型形式化**（显式状态对象） | `practice §模式七` |
| 过大模块（8） | **分离关注点**（`posters.py` 承接制品渲染） | `practice §模式二` |
| 重复代码（9） | **删除死路径 / 去重**（仅删可证逐字冗余） | `practice §模式六` |
| 历史配置（10） | **归档**（非破坏性 `git mv`，保留 commit 历史） | `practice §安全守则` |
| 上帝类（12） | **按职责拆分**（mixin 转发，保持权重键） | `practice §模式二、五` |
| 正确性（11） | 逐项修复 + **黄金测试**固化 | 审计 §2、§8 |

### 决策说明

1. **为什么用「连续区段抽取」而不是「按语义重排」？**
   两个 CLI 的 `add_argument` **顺序本身是语义**（`--help` 输出、位置参数顺序、默认值覆盖）。因此只抽**连续行区间**，并在抽取后以 **actions 指纹**（逐 action 的 `option_strings/dest/type/default/choices/nargs/required/help`）**逐字比对**——实测 247（评测）与 334（训练）个 actions **完全一致**。这比 `--help` 文本比对更强（`--help` 会被 help 串里的 `%` 打成 `TypeError`）。

2. **为什么用 mixin 而不是审计示意图里的新层级（`self.backbone` / `self.conditioner`）？**
   审计 §4.1 明确警告：**"不能直接按这种新层级替换现有模型：那会改变全部 `state_dict` key"**。mixin 拆分只改变方法的定义位置，**不改变 `self.xxx` 属性名与注册顺序**，因此 `state_dict` 键逐字不变——由**前向黄金**与**梯度黄金**双重裁决。

3. **为什么"逐字搬移 + 顶部重绑"？**
   被搬移的块**一字不改**，只在目标函数顶部把自由名重绑为参数/上下文。等价性是**构造性**的，不依赖逐行人工比对；再由逐位黄金验证。

4. **为什么把 `EvalSetContext` 做成 dataclass 而不是 22 个位置参数？**
   位置参数在调用点无法自证顺序，且审计要求"显式状态"。dataclass 使字段名即文档，且**全部只读**。

5. **为什么只删"逐字相同"的副本，不删"相似但不同"的？**
   审计 §5 明确："这不是已经证明运行时完全不可达的文件名单"。`tools/legacy/` 经实测是**真归档**（187 个中逐字相同仅 11、旧版本 57、只在 legacy 119）——只有那 11 个是零信息冗余。

6. **被否决的方案：**
   - **按语义重组 `add_argument`**：会改变顺序 → 否决。
   - **用 `self.backbone` 新层级替换模型**：会改 `state_dict` 键 → 否决（审计同款警告）。
   - **按"零引用"删除 `tools/` 脚本**：`tools/` 是**手工入口**，零引用 ≠ 未使用 → 否决。

---

## 5. 变更明细

### 5.1 训练主函数：从 2,429 行到 1,549 行的生命周期

**文件：** `src/train/train.py`、`src/train/trainer.py`（新增）

**修改前：**
```python
def main(args):
    # ... 2,429 行：模型/数据/DINO/Critic/正则/EMA/编译/探针/恢复/保存/评测 ...
```

**修改后：**
```python
def main(args):
    # ... 配置解析、组件装配 ...
    _train_fatal = _training_loop(_TrainState(args=args, ...))
```

**说明：** 810 行的训练循环**逐字搬进** `trainer.py:training_loop(state)`，状态收进 `@dataclass TrainState`（61 字段）。另抽出 `step.py`（`resolve_step_inputs` / `clip_and_step` / `build_train_model_kwargs`）、`telemetry.py`（`StepProfiler` / `Diagnostics` / `WindowAverages` / `format_step_log`）、`checkpoint.py`（`CheckpointManager`）、`optimizer.py`、`ema.py`、`backward.py`、`objectives.py`。

### 5.2 模型构造器：`__init__` 870 → 481 行（mixin 转发）

**文件：** `src/model/dit.py`、`src/model/dit_condition_mixins.py`（新增）

**修改前：**
```python
class DiT_2Cond(nn.Module):
    def __init__(self, ...):          # 870 行
        # Transformer block + 书家/书体/字符表 + STD 编码 + 条件路由融合
        # + 空间风格调制 + Skeleton/Deform 遗留 + CFG + checkpoint 兼容
```

**修改后：**
```python
class DiT_2Cond(DitConditionMixin, nn.Module):
    def __init__(self, ...):          # 481 行
        ...
        self._init_conditioning_block(...)   # 338 行块 → 方法
        self._validate_content_cond(...)     # 纯校验
        _sl = self._resolve_style_ada(...)   # 风格支路层解析
```

**说明：** 4 个构造器方法（`_build_glyph_condition` 133 / `_build_style_hierarchy` 115 / `_build_deform` 30 / `_build_cond_router` 21）搬进 mixin；`__init__` 内再切三刀（338 行条件块、29 行纯校验、25 行风格解析）。**`state_dict` 键与注册顺序不变**。

### 5.3 评测编排：`run_in_mem_eval` 388 → 106 行

**文件：** `src/eval/in_mem_eval.py`

**修改前：**
```python
def run_in_mem_eval(...):
    # ... 283 行 Try：采样 + 解码 + 指标 + 制品 + CSV 全在里面 ...
```

**修改后：**
```python
def run_in_mem_eval(...):
    ...
    _ = _eval_one_set(EvalSetContext(cfg=cfg, cks=cks, dev=dev, ...))

def _eval_one_set(ctx: EvalSetContext):   # 317 行，逐字搬移
    ...
```

**说明：** 283 行块抽成 `_eval_one_set`，外部状态收进 `@dataclass EvalSetContext`（28 字段）。`run_in_mem_eval` 现在只做编排。

### 5.4 评测 CLI：`build_parser` 866 → 32 行（8 个连续参数组）

**文件：** `src/eval/cli.py`

**修改前：**
```python
def build_parser(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(...)   # ×253 条，平铺 866 行
```

**修改后：**
```python
def build_parser(argv=None):
    parser = argparse.ArgumentParser()
    _add_args_1(parser); _add_args_2(parser); ... _add_args_8(parser)
    ...
```

**说明：** 按注释分段切成 8 个**连续**区段（79/257/53/59/217/107/54/30 行）。**actions 指纹逐字一致**（247 个）。

### 5.5 训练 CLI：`build_parser` 1,193 → 77 行（9 个连续参数组）

**文件：** `src/train/cli.py`

**说明：** 同 §5.4，9 个区段（93/316/53/75/237/104/168/58/37 行）。**actions 指纹逐字一致**（334 个）。两轮合计 `build_parser` **2,059 → 109 行**。

### 5.6 海报与评测助手归位

**文件：** `src/eval/posters.py`（275 → 762 行）、`src/eval/in_mem_eval.py`

**说明：** `render_poster`(219) + `_poster_canny`(17) + `_poster_skeleton`(14) 及 6 个助手（`_steps_scan` / `_train_ref_rows` / `_train_nn_row` / `_load_font` / `_step_ssim_txt` / `_font_cache`，共 ~190 行）搬进 `posters.py`。原处**再导出** ⇒ 28 处调用点一字不改（实测 `in_mem_eval.render_poster is posters.render_poster`）。曾出现**循环导入**，以函数内惰性导入临时破环；随后把助手**整体搬走**，彻底删除该惰性导入。

### 5.7 配置与冗余归档（非破坏性）

**文件：** `src/train/configs/`、`tools/legacy/`、`tools/`

**说明：** 先做**引用扫描**（审计 §5 规定的流程），再执行：

| 对象 | 处理 | 依据 |
|------|------|------|
| 200 个**零引用**历史配置 | `git mv` → `configs/archive/` | git 全部识别为 **R 重命名（200/200）**，历史保留 |
| 11 个 `tools/legacy/` 逐字副本 | `git rm` | sha256 相同 + 零显式引用 ⇒ 零信息损失 |
| 12 个 `tools/` 内逐字副本 | `git rm` | 同上（保留被引用/顶层/短路径者） |

---

## 6. 验证结果

| 验证项 | 结果 | 备注 |
|--------|------|------|
| 单元测试 | ✅ | **434 passed**（含 27 个重构专项测试文件） |
| Linter | ✅ | `ruff check src tests tools/prof/bench_train.py` → 0 条 |
| 静态未定义名 | ✅ | `pyflakes src` → 0 条（`tools/preflight.py --strict` FAIL=0） |
| **前向黄金（逐位）** | ✅ | `tools/verify_forward_parity.py` → exit 0，**逐位一致** |
| **梯度黄金（逐位）** | ✅ | `tools/verify_grad_parity.py` → **逐位一致**（59,818,645 个梯度元素） |
| 训练 smoke | ✅ | `tools/smoke_train.py --steps 2 --device cpu` 通过 |
| 精确恢复 | ✅ | `tests/test_refactor_resume.py`（5 项状态逐位一致，含数据顺序） |
| CLI 等价 | ✅ | actions 指纹逐字一致（评测 247 / 训练 334） |
| 配置有效性 | ✅ | `tools/audit_configs.py --active --strict` → 14 个全部有效 |

**已修复的审计项**：P0-1/P0-2/P0-3、P1-1～P1-7 共 **11 项全部清零**。

---

## 7. 后续建议

### 7.1 仍待完成（需 GPU 空闲）

- **PR6 基准实测**：工具已就绪并**核实**（`tools/prof/bench_train.py`），等 v88 训练结束即可跑：
  ```bash
  python tools/prof/bench_train.py --benchmark A --batch 192,240,288 --steps 30
  python tools/prof/bench_train.py --matrix dataloader --batch 288 --steps 30
  ```
- **PR3 遗留**：`eval_auto.eval_gen_in_memory` 迁到共享采样器，需 **GPU 逐位对比**。

### 7.2 需人工处理

- `src/eval/batch_eval.py:main`（**368 行**）：自动切分未收敛（105 个局部 + 块内循环变量），已回滚，需人工拆分。

### 7.3 审计范围之外的重要发现

- ⚠ **审计 §1 的规模表只覆盖 `src/`**。实测 `tools/`（递归）= **1,272 个 / 107,675 行**，是 `src/`（40,493 行）的 **2.8 倍**——**审查漏掉了最大的一块**。建议对 `tools/` 单独立项治理。
- ⚠ `prefetch_factor` **未暴露为 CLI 开关**，因此审计 §7.2 矩阵的 **D(8,2) / E(8,4) 两行当前无法从命令行区分**，需先在配置中暴露该键。
- ⚠ `tests/test_c2ot.py::test_c2ot_zero_prior_shift` 存在**既有 flaky**（单独跑通过、复跑通过），与本次重构无关，建议单独立项排查。

### 7.4 已确认无需改动

- `src/model/dit_conditioning.py`（1,337 行）：**函数最大仅 238 行**，且已有 8 个显式 `*Cfg` dataclass——**已达审计 §4.1 的目标（100–250 行/模块）**。

---

## 附录：工程纪律（本轮新增，供后续复用）

1. **等价性优先用逐位黄金裁决**，而不是人工比对——本轮所有结构性改动都由前向/梯度双黄金放行。
2. **共享仓库禁用目录级 `git add`**（本轮误扫他人文件 3 次，均当场纠正）——一律显式列文件，提交前核对暂存清单。
3. **非破坏性归档优先于删除**：`git mv` 保历史；只删**可证逐字冗余**。
4. **用工具的真值替代自算**：静态分析器会漏嵌套作用域定义，改用 `pyflakes` 驱动补齐。
5. **移动代码后必须按目标容器重新缩进**（函数体 4 / 类体 4）——本轮因此类失误 3 次，均由即时验证暴露。
