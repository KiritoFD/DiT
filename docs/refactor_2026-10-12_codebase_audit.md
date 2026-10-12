# 重构报告：Callig-DiT codebase 巡检与加固（2026-10-12）

> 触发：v81 白化事故排查中暴露出一批**同类隐患**（静默默认、未绑定变量、约定常数蔓延）。
> 原则：**重构 ≠ 重写**；只做**行为中性**或**可静态证明可达性不变**的改动；接口全部兼容。

## 一、问题清单（按"是否真实发生过事故"排序）

| # | 坏味道 | 位置 | 事故史 | 处置 |
|---|---|---|---|---|
| 1 | **getattr 兜底默认 ≠ 配置共识** | 46 处（`src/eval/in_mem_eval.py:906`、`src/model/train.py:288/298/417`、`src/train/train.py:497/513/543/1237` …） | E5（sf 抄 0.18215 未随 VAE 重拟合，潜伏 3 周） | **P2 显式化 + 闸门硬门槛** ✓ |
| 2 | **分支内赋值 / 外层引用** | `src/train/train.py:1772`（`_repa_cache_dir`）| 2026-10-12 回归：`w_repa=0` 必崩 ✗ | 自足表达式（他处已修）+ **pyflakes 纳入闸门** ✓ |
| 3 | **未定义名**（用了没赋值） | `src/eval/auto_eval_cpu.py:85`、`src/eval/gpu_ablate_eval.py:205` | 潜在崩溃（评测路径 ✗）| 定点修复（删死代码 / 恢复返回值）✓ |
| 4 | **约定常数蔓延** | `0.18215` **460 处**、`TIME_SCALE` **136 处** | E5 扩散根因 | **棘轮锁死**（只许下降）+ 后续收敛为单一来源 |
| 5 | **模块级单槽缓存** | `in_mem_eval.py`: `_VAE/_DIFF/_CMAP/_LPIPS_FN/_CSMAP`；`modules.py: _XOPS` | E3（按 device 缓存 VAE）/E9（就地打补丁）| `_VAE` 键已补全（路径+shift）；其余 5 个待核 ⬜ |
| 6 | **不可达死代码** | `auto_eval_cpu.py` 3cond 分支（DiT_3Cond 已删）| — | 删 14 行 ✓ |

## 二、改动清单（全部行为中性）

| 文件 | 改动 | 依据 |
|---|---|---|
| `tools/preflight.py` | **新增**：统一启动闸门（工具自检 + pyflakes + 常数棘轮 + 配置显式性）| 这次事故的通用化 |
| `tools/preflight.py`（含 H3 规则）| **新增并成为唯一入口**：`validate_config.py` 与 `static_check.py` 完成后即删除（前者规则并入 preflight，后者被 pyflakes 取代）| 46 处 H3 |
| `tools/analysis/audit_repo.py` | **新增**：6 类"已发生过的事故"可扫描 | 巡检 |
| `tools/analysis/static_check.py`、`white_baseline.py`、`code_md5.py`、`v81_faintness.py` | **新增**：静态检查 / 指标白图零点 / 两端一致性 / 墨量体检 | 本次复盘沉淀 |
| `scratch/check_v83_wiring.sh` | **新增**：接线熔断（缺失即 kill）| 实战拦下一次错跑 ✓ |
| `src/eval/auto_eval_cpu.py` | 删 raise 之后 14 行不可达代码 | 可静态证明不可达 ✓ |
| `src/eval/gpu_ablate_eval.py` | `miss, unexp = model.load_state_dict(...)` | strict=True 成功即空表，日志恢复真实 ✓ |
| `tools/build_fame_fast.py`、`tools/data/build_fame_fast.py` | 补 `import glob` | NameError 修复 ✓ |
| `v83A/v83B_stdsoft_zerofix.json` | 9 个隐式默认键**按当前生效值**显式写入 | 重启后逐位一致 ✓ |

## 三、验证

```
python tools/preflight.py --config <cfg> --strict      # FAIL=0  exit=0 ✓
  ✓ pyflakes 自检 (4.0.3)
  ✓ pyflakes src/            0 条（原 3 条）
  · pyflakes tools/+legacy/  47 条 → 棘轮锁死
  ✓ 配置显式性(H3)           0 项（原 9）；全仓高危 2779 → 621
  ✓ 棘轮 0.18215 460→460 · TIME_SCALE 136→136
语法自检: 4/4 文件 OK ✓        行为对比: 仅新增键，无数值变更 ✓
```

## 四、约束与后续（未做，按性价比排序）

| 优先级 | 项 | 说明 |
|---|---|---|
| P0 | **把 `preflight --strict` 接进每次启动** | 闸门不接进流程 = 不会被执行 |
| P0 | **`sf/shift` 收敛为单一来源**（`src/utils/vae_scaling.py`）| 460 → 1；棘轮保证不倒退 |
| P1 | 核 `_DIFF`（γ/steps/cfg 若不在缓存键里 → 换 γ 仍复用旧解算器 ✗）等 5 个缓存 | E3/E9 同类 |
| P1 | 收紧全仓 621 项配置显式性（按"本 run 真正读取的键"逐个显式化）| 建议跟随"下次改到该配置时"顺手做 |
| P2 | H1 启发式（269 噪声）升级为真控制流分析 | pyflakes 已覆盖真信号，优先级低 |

## 五、经验（写给未来的自己）

1. **工具的"通过"必须先证明工具真在跑** —— 我曾把"pyflakes 未安装"报成"0 处可疑" ✗。
2. **指标必须过"零点基线"** —— SSIM 给全白图 0.6855，高于模型真实输出 0.6371 ✗ → 该类指标永久降级为参考项。
3. **启动前必须过接线/配置闸门** —— 本轮它拦下了一次跑错配置的 200k 实验 ✓。
4. **重构用小步、可验证、可回退的提交** —— 本次 2 个提交，每步都有静态验证 ✓。

---

## 六、第二轮（同日）：审查 → 精简 → 文档同步

**审查发现（全部为"已损坏/必崩"的真实问题，非风格问题）**

| 类别 | 位置 | 证据 |
|---|---|---|
| **全局替换事故污染标识符** | `sample_controlnet.py`(×2)、`make_fame3_poster.py`、`_diag_skel_similarity.py`、`legacy/gradio/gradio_fame_local.py` | `std_skel` → `data/skel/std_skel` 命中**标识符**：`def render_data/skel/std_skel(` 直接无法解析 ✗ |
| **mojibake 吃掉换行** | `sample_controlnet.py`(×2) | 中文注释被 GBK↔UTF-8 双重解码碾平，`skel_path = ...` 整行落进注释 → L174 **NameError** ✗ |
| **UTF-8 BOM** | 10 个文件 | 噪音；且让"按 str 编译"的检查误报（`compileall` 读 bytes 所以不报 ✗）|
| **裸转义** | 4 个 tools | `\d` `\|` `\p`（未来 Python 变 SyntaxError）|

**精简**

| 动作 | 对象 | 理由 |
|---|---|---|
| 删 | `validate_config.py` | H3 规则 100% 被 `preflight.py` 覆盖 |
| 删 | `static_check.py` | 被 pyflakes + 语法门取代（噪声 295 vs 真信号 0）|
| 删 | `apply_refactor_p1.py` / `p2.py` | 一次性迁移脚本，已完成（git 历史保留）|
| 删 | `sde_oracle_test.py` / `2.py` | 迭代中间版；保留 `3.py`（被 `src/loss/flow_matching.py:501` 注释引用 ✓）|
| 去后缀改名 | `fix_sweep_images2.py`、`make_ckpt_sweep_poster2.py` | 去掉"2"，消除编号重复味道 |

**闸门补强**：新增 **语法门**（`src/` 零不可编译为硬门槛）。
> 必要性：pyflakes **只在文件可解析时才有信号** ✗ —— 上述 5 个语法损坏文件被它整体跳过，
> 旧的闸门因此全绿而实际已有文件损坏 ✗。**语法门必须排在 pyflakes 之前** ✓。
> 有效性已证明：故意投放坏文件 → 立刻 FAIL ✓。

**验收**

```
python tools/preflight.py --config src/train/configs/v83A_stdsoft_zerofix.json --strict
  ✓ 语法 src/ 0   ✓ pyflakes src/ 0   ✓ H3 0   ✓ 棘轮 460/136 持平      FAIL=0 exit=0
```

**新增教训（补进第五节）**

5. **读文件一律 `utf-8-sig`**，与解释器对齐；否则 BOM 制造假阳性 ✗。
6. **`compileall(quiet=1)` 会吞掉错误输出** —— 我用它做的"编译 OK"是**假绿** ✗，改用逐文件 `compile()` ✓。
7. **一次全局替换的杀伤半径要用"全仓 grep + 编译门"双重兜底** ✗。
