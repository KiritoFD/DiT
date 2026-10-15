# Callig-DiT 重构进度报告（PR0–PR4）

**日期**: 2026-10-16（凌晨）
**范围**: 审查报告《Callig-DiT 全面代码审计与单卡 4090 重构方案》的 **PR0 / PR1 / PR2 / PR3(部分) / PR4(部分)**
**HEAD**: `221323f` · 提交数 733
**审计对象**: 本仓库当前 HEAD（审查的 `src.rar` 缺陷已**逐处复现**后修复 ✓）

> ⚠ **一处诚实说明**: 本会话早前若干提交信息里把日期写成 `2026-10-14` ✗，
> 实际提交时间是 **2026-10-15/16** ✓（`git log --format=%ci` 为准 ✓）。仅信息笔误 ✓ 内容不受影响 ✓。

---

## 1. 一句话结论

审查列的 **11 项 P0/P1 正确性缺陷已全部逐处核实并修复** ✓（含审查自己复现的两项 ✓），
**PR2 全部完成** ✓，**PR4 完成四个可独立验证的切片** ✓，
**PR3 只剩一项需要 GPU 逐位对比才能安全推进** ✓（4090 正被 v88 预训练占用 ✓ 已记录待办 ✓）。

**验证体系**: 12 个重构测试文件 / **299 项测试全绿** ✓ + 前向黄金 **逐位一致** ✓（全程保持 ✓）。

---

## 2. PR1：正确性缺陷（11 项全部完成 ✓）

| 编号 | 缺陷（审查原文要点） | 修复 | 关键证据 |
|---|---|---|---|
| **P0-1** | 评测完成协议**从未接入主流程** ✗；`is_complete` 只校验"已登记"产物 ⇒ **伪造空 DONE 也判完整** ✗（审查已复现 ✓） | ① `protocol.py`: manifest 必须存在 ✗ + PNG 数**必须恰好 == n_samples** ✗ + **原子写** ✓ + `is_complete` 三道**磁盘重算**检查 ✓ ② **接入 `in_mem_eval`** ✓：身份 = **权重内容 sha256** ✓（in-training eval 无 ckpt 文件 ✗）+ **DONE 存在即权威** ✓（校验失败 ⇒ **强制重跑** ✗） | `tests/test_refactor_p0.py` 16 项（含审查最小复现 ✓）· `test_refactor_p4.py` 9 项（含**真实 187 张 PNG** 往返 ✓） |
| **P0-2** | `np.empty` 预加载 ⇒ **未初始化内存进 batch** ✗（注释还谎称"会是全 0" ✗） | 5 处 → `np.zeros` ✓（让那句承诺成真 ✓） | `test_refactor_p1.py` 静态守卫 ✓ |
| **P0-3** | Critic 查表 `np.clip` ⇒ **缺失静默映射到第 0 行** ✗（生成「家」却喂别的字的 STD ✗） | 严格查表 ⇒ `KeyError/IndexError` + 缺失样本清单 ✗ + `coverage_audit()` ✓ | `test_refactor_p0.py` 5 项 ✓ |
| **P1-1** | 前向中**创建参数** ✗（optimizer 之后新建 ⇒ 参数不进优化器 ✗） | 构造期定型 ✓ + `__init__` 末尾**锁定** ✓ ⇒ 锁定后报错 ✗ | **前向黄金 bit-exact** ✓ + 4 项测试 ✓ |
| **P1-2** | ckpt **不存 RNG** ✗ ⇒ "可续训但**不可逐样本重放**" ✗ | `capture_rng/restore_rng` 四路状态 ✓ + `rng_meta{epoch_aligned, resume_kind}` ✓ + 循环前恢复 ✓ | `test_refactor_p2.py` 8 项（**捕获→消耗→恢复⇒抽样逐位一致** ✓）|
| **P1-3a** | 异步 ckpt 失败只打印 ✗ ⇒ "训练完成但模型没存" ✗ | `drain_ckpt(strict=默认开)` ⇒ 失败**抛错** ✗ ⇒ 非零退出 ✓ | `test_refactor_p1.py` 4 项 ✓ |
| **P1-3b** | `freeze_table()` **最多调 3 次** ✗ | 收敛为唯一一次 ✓ | 静态核对 ✓ |
| **P1-4** | 注释谎称 npz+mmap "只读 slice" ✗（审查实测无效 ✗） | 注释纠正为"每次解压整个 shard" ✓ + 指明 PR2 正解 ✓ | 静态核对 ✓ |
| **P1-5** | EMA 注释谎称"严格等价" ✗ | 纠正为**近似** ✓ + interval 入实验身份 ✓ + **偏差量化工具** ✓ | `test_refactor_pr4c.py` 22 项 ✓ |
| **P1-6** | 每步 CUDA 同步 ✗（critic 三处 + strong_style 逐样本 ✗） | critic 三处 ⇒ detached tensor ✓；strong_style 资格筛选**向量化**（B 次同步 → 1 次 ✓）+ g 量纲只判一次 ✓ | `test_refactor_pr4.py` 7 项（**与逐字复刻的旧循环 25 组等价对比** ✓）|
| **P1-7** | 清单只哈希 `src/**/*.py` ✗ ⇒ **无法证明用了哪版 Critic** ✗ | 多根（src + tools ✓ 实测 **Critic 已在清单内** ✓）+ 权重/VAE/字体/数据各自 SHA256 ✓ + 缺失**显式记录** ✗ | `test_refactor_p3.py` 8 项 ✓ |

**两处「比审查更细」的发现** ✓
- **P1-6**: 三处"逐样本同步"里**只有两处是真的** ✗ —— `torch.randint(len(valid), ())` **无 device ⇒ CPU 张量** ✓ 其 `.item()` 是 **CPU 侧** ✗ **不是** CUDA 同步 ✓；且它带 RNG ⇒ 改它会**改变随机流** ✗ ⇒ **有意保留** ✓
- **dino_cache**: 另发现一处**静默 dtype 差异** ✗✗ —— pinned 路径不做 `.float()`（返回磁盘 dtype ✗）而 memmap 路径做 ✓ ⇒ **同一 cache 因内存是否够而返回不同 dtype** ✗ ⇒ 已统一为**恒返回 fp32** ✓

---

## 3. PR2：数据层（完成 ✓）

| 项 | 修复 | 证据 |
|---|---|---|
| **LatentContract** | 发现**真实覆盖缺口** ✗：`_check_shard_names` 只在 `got is not None` 时比对 ⇒ **shard 完全没有该 id 时静默通过** ✗✗ ⇒ 新增 `strict_coverage=True` ✓ `audit_coverage()`（逐族核对 **CSV 每一行** ✓ 缺行 ⇒ **启动前报错** ✗）+ `latent_contract()` 身份（rowmap_sha256 ✓）| `test_refactor_pr2b.py` 8 项 + **真实 shard 双向验证** ✓（7460 条 id 全覆盖通过 ✓；3 个不存在的 id ⇒ 报错并列出 ✗）|
| **REPA 缓存检查** | ① 文档谎称 async H2D ✗（代码自己承认是**同步** ✓）⇒ 如实改写 ✓ + 指出真异步需**双缓冲/CUDA Event** ✓ ② **消除静默 fallback** ✗（pinned 失败默认报错 ✓）③ 修**静默 dtype 差异** ✓ ④ `require_coverage()` ✓ ⑤ `ram_dtype="fp16"`（**默认关** ✓ 需实测 ✓）| `test_refactor_pr2.py` 14 项（**磁盘 fp32/fp16 × pinned/memmap 四组合均返回 fp32** ✓）+ `tools/_smoke_dino_cache.py` **SMOKE PASS** ✓ |

---

## 4. PR3：单一采样器（主体验收通过 ✓，剩 1 项待 GPU ✓）

**关键核查结论** ✓: "单一采样器"**早已就位** ✓ —— `src/eval/sampler.py` 的 `sample_latents` 已被
`in_mem_eval` / `batch_eval` / `eval_ctrl_ckpt` 采用 ✓，且 `inference.py` 是**同一函数对象**的兼容再导出 ✓
（实测 `inference.sample_latents **is** sampler.sample_latents` ✓）。

**已退役** ✓（按审查 §5 删除流程逐步取证 ✓）: `gpu_batch_eval.py` / `gpu_batch_eval_v2.py`
→ `git mv` 到 `src/eval/legacy/` ✓（**R097 重命名 ⇒ git 历史完整** ✓）
退役依据: 真实 `import` = **0** ✓ / `*.json` 引用 = **0** ✓ / 外部工具调用 = **无** ✓ / 动态 import 通过但无人调用 ✓。

**⚠ 剩余 1 项（已如实记录，等 GPU）** ✗
`src/eval/auto_eval_cpu.py`（**活跃** ✓ 7 处配置引用 ✓）→ `eval_auto.eval_gen_in_memory` ✗
= **自建 diffusion + 自跑采样循环** ✗ ⇒ **活跃路径绕过了单一采样语义** ✗
⇒ 迁移**必须**先做 GPU 逐位对比（PR3 完成标准: "与旧版相同的固定采样结果" ✓）
⇒ 4090 被 v88 占用期间**不动它** ✗（改采样却无法验证 = 违反 PR3 标准 ✗）
⇒ 已写入 `test_refactor_pr3.py` 的 `KNOWN_LIVE_SELF_IMPL` ✓（**新增**重复路径会失败 ✗；已知项**消失**也失败 ⇒ 提醒同步 ✓）

**一处自我纠错** ✓: 我一度把 `ddim_sample_loop` 当作"遗留 DDPM 标记" ✗ —— 它是**共享采样器的统一接口** ✓
（`sampler.py` 自己也调用它 ✓）⇒ 已订正归档说明与测试措辞 ✓（归档决策依据不变 ✓）。

---

## 5. PR4：训练路径收敛（4 个切片完成 ✓）

### 新增模块边界（对照审查 §3 目标结构 ✓）

| 审查目标模块 | 本仓库落点 | 内容 |
|---|---|---|
| `training/step.py`（backward 策略 ✓）| **`src/train/backward.py`** ✓ | `joint_backward` / `sequential_backward(loss, aux_factory)` / `aux_backward` / `grad_norm` / `resolve_strategy` / `describe` / `param_groups_for_probe` |
| `training/ema.py` ✓ | **`src/train/ema.py`** ✓ | `update_ema`（逐字搬移 ✓ + 再导出 ✓）/ `should_update` / `effective_decay` / `interval_approximation_error` / `describe` |
| `training/objectives.py` ✓ | **`src/train/objectives.py`** ✓ | `ObjectiveComposer` / `compose` |
| （沿用既有 ✓）| `src/train/ckpt.py` / `setup.py` / `data_setup.py` / `gradient_budget.py` / `manifest.py` | 审查已认可的既有拆分 ✓ |

### 切片 1：消除逐样本 CUDA 同步（P1-6 收尾 ✓）
资格筛选**向量化**（每步同步 **B 次 → 1 次** ✓）+ `g` 量纲只判一次 ✓。
★ 等价性测试抓到**真实边界语义差异** ✗✗: `t=0.8` 存成 float32 = `0.8000000119` ✗ ⇒
旧代码 `float(t[i]) <= 0.8` 是 **float64** 比较 ⇒ **排除** ✓；若用 float32 张量比标量 ⇒ **包含** ✗
⇒ 必须 `.double()` ✓（且**我最初的测试期望也写错了** ✗ ⇒ 已订正 ✓）。

### 切片 2：backward 策略命名并集中 ✓
★ **顺序本身就是语义** ✗: `sequential_backward` 用 **`aux_factory` 回调**把
"先释放 FM 图、**再**构造辅助 loss"变成**类型层面无法写错**的约束 ✓✓
（提前构造 ⇒ 两张图同时存活 ⇒ 完全失去省显存意义 ✗）。
`train.py` 中**裸 `loss.backward()` 归零** ✓（守卫测试锁定 ✓）。
★ 自我纠错: 首版顺序探针用 `loss.grad_fn is None` 判断"已反传" ✗ —— backward 后 `grad_fn` **仍非 None** ✗
（只有 saved tensors 被释放 ✓）⇒ 假阴性 ✓ ⇒ 改为两侧自定义 `autograd.Function` 打点 ✓ 实测
`events == [main_backward, factory_called, aux_backward]` ✓✓。

### 切片 3：EMA 组件化 + 间隔偏差量化 ✓
★ **量化结果**（审查 P1-5 的定量依据 ✓）: 单窗口注入偏差（β=0.999）
`N=1 ⇒ 0` ✓ `N=4 ⇒ 0.0060` ✓ `N=8 ⇒ 0.0279` ✓ `N=16 ⇒ 0.119` ✓；
但实测发现**更重要的是累积效应** ✗: 偏差每 N 步注入且**同号** ⇒ 总差**线性增长** ✓
（200 步 ⇒ 0.27 步漂移 ✓；2000 步 ⇒ 1.30 ✓）⇒ 间隔 EMA 有**系统性滞后** ✓ **不是**零均值噪声 ✗
⇒ 这就是"**不是**数学等价"的定量证据 ✓ 换 interval = 换 EMA 语义 ✓。
★ 自我纠错: 搬移 `update_ema` 时**丢掉了 `@torch.no_grad()`** ✗✗ 且留下**孤儿装饰器** ⇒ SyntaxError ✗
⇒ 已补回 ✓ 并把"装饰器必须随函数一起搬"写进测试 ✓。
★ 自我纠错: 首版把闭式误差当"总差" ✗ ⇒ 与多步模拟差 45 倍 ✗ —— **闭式没错** ✗ 而是**定义不同** ✓
⇒ 已精确化文档 + 改为单窗口精确对照（<1e-12 ✓）+ 新增累积效应测试 ✓。

### 切片 4：ObjectiveComposer（损失装配）✓
★ 两条写进代码与测试的**不变量**:
① **注册顺序 = 求和顺序 = 语义** ✗（浮点加法不满足结合律 ✓ 实测同组数乱序 ⇒ `0.0` vs `1.0` ✓）
② 零权重项**照样加** ✗（`0.0*nan = nan` ⇒ 坏样本 NaN **照实传播** ✓ 不被静默吞掉 ✓）
★ raw 一律 **detached tensor** ✓ ⇒ 唯一 CUDA→CPU 转换点是 `as_floats()` ✓（只在日志周期调用 ✓ 审查 P1-6 ✓）
实测: 同一组随机张量下 `total` 与手写链 **`torch.equal` 逐位一致** ✓。

---

## 6. 验证体系（可复现 ✓）

```bash
# 全量测试（388 项 ✓）
python -m pytest tests/ -q

# ★★ 梯度黄金（审查 PR0 明确要求 ✓ 2026-10-14 补齐 ✓）
python tools/verify_grad_parity.py --ckpt _local_ckpt/0055000.pt \
    --golden assets/_audit/grad_v84A55k.json --check

# 前向黄金（**逐位一致** ✓ 全程保持 ✓）
python tools/verify_forward_parity.py --ckpt _local_ckpt/0055000.pt \
    --golden assets/_audit/fwd_v83A55k.pt --check

# 训练链路冒烟
python tools/smoke_train.py --config src/train/configs/v84A_std_cond.json --steps 2 --device cpu

# REPA 缓存冒烟（gather / 缓存路径 / miss 回退 ✓）
python tools/_smoke_dino_cache.py

# 环境与配置门禁
python tools/preflight.py --strict
python tools/audit_configs.py --active --strict
```

### 6.1 ★★ 梯度黄金（本轮补齐 ✓ 审查 PR0 的硬要求 ✓）

审查 PR0 原文:
> **梯度等价测试** —— 在固定 FP32 小 batch 上，检查**主干和条件层**的梯度，
> 避免"前向看起来正常，实际训练方向变了"。

**为什么必须单独做** ✗: 既有前向黄金跑在 `torch.no_grad()` 里 ✓ ⇒ **管不到反传** ✗；
而 PR4 的搬移**大量发生在反传与更新路径** ✓（backward 策略 ✓ 损失装配顺序 ✓ clip/EMA ✓）
⇒ 只有前向黄金**无法**证明"训练方向没变" ✗。

**工具**: `tools/verify_grad_parity.py` ✓ 记录
① 全参数梯度按**固定顺序**拼接后的 **sha256**（逐位 ✓）
② 按角色分组的梯度范数（backbone / condition ✓ 便于**定位**是哪一组变了 ✓）
③ 固定函数值 ✓

**实测（v84A 55k ✓ 本机 CPU ✓ FP32 ✓ batch=2 ✓）**:

| 项 | 值 |
|---|---|
| 有梯度的参数数 | 160 ✓ |
| 梯度元素总数 | **59,818,645** ✓（≈ 审查的 P=59.9M ✓ **交叉吻合** ✓）|
| `backbone` ‖g‖ | 5.653889e+03 ✓ |
| `condition` ‖g‖ | 2.012861e+03 ✓ |
| `grad_sha256` | `4270a845da278c6da2a92155f5e592810c82e79ae491d48c993acd0a9fa34f36` ✓ |

**确定性** ✓: 连跑两次 sha **完全相同** ✓（否则黄金无意义 ✗）。

**★★ 阴性对照**（证明黄金**会失败** ✓ 不是假测试 ✗）:
微扰**单个** 2D 权重 `+1e-6` ⇒ sha 变为 `0b7325f2…` ✓（`backbone` 范数变化 1.86e-3 ✓）
⇒ 灵敏度**实测确认** ✓（该对照已固化为测试 ✓ `test_golden_is_sensitive_to_tiny_perturbation` ✓）。

**设计取舍** ✓: 损失用**固定随机泛函** `(out·W).sum()`（W 由固定 seed 生成 ✓）
而非完整 FM 损失 ✓ ⇒ 黄金**不依赖** diffusion / REPA / 数据集的内部实现 ✓
⇒ 它们各自的搬移不会**误报** ✓；而模型的**前向 + 反传 + 条件层梯度**仍被**逐位**锁定 ✓
（这正是 PR4 / PR5 要保护的东西 ✓）。

### 6.2 ★★ 精确恢复测试（本轮补齐 ✓ 审查 PR0 第 3 条 ✓）

审查 PR0 原文:
> **精确恢复测试** —— 在固定环境下比较**连续训练 100 步**与**训练 50 步、保存、
> 恢复后再训练 50 步**。**数据顺序、参数、optimizer、EMA、scheduler** 应符合
> 明确的恢复契约。

★ 这同时是 **PR4 的完成标准**（审查 §8: "PR4 … 完成标准 = **Exact Resume 可重放**" ✓）

**测试**: `tests/test_refactor_resume.py` ✓（**5 项** ✓）
在一个最小可训练模块上跑**完整更新**（数据顺序 → 前向 → 反传 → clip → step →
scheduler → EMA ✓），比较**两条路径**:
* 路径 A: **连续 100 步** ✓
* 路径 B: **50 步 → 保存 → 序列化往返 → 恢复 → 50 步** ✓

**判据 = 逐位** ✓（审查要求"逐步、逐样本可重放" ✓ 只允许 `torch.equal` ✓）:

| 状态 | 检查 |
|---|---|
| **数据顺序** | 每步抽样的**样本索引序列**必须完全一致 ✓（**最灵敏的探针** ✓）|
| 参数 | 全部张量 `torch.equal` ✓ |
| optimizer | `exp_avg` / `exp_avg_sq` / `step` ✓ |
| EMA | 全部张量 `torch.equal` ✓ |
| scheduler | `last_epoch` ✓ + `lr` ✓ |
| RNG | `torch.get_rng_state()` 逐位 ✓ |

**★ 探针为什么灵敏** ✓: 每个 micro-step 用 `torch.randint` 抽样本索引 ✓
（**消耗 torch RNG** ✓）+ 模块里带 `Dropout` ✓（前向本身也消耗 RNG ✓）
⇒ 只要 RNG 没恢复 ✓ **数据顺序立刻不同** ✓ 无需等数值漂移 ✓

**★★ 两处自我纠错** ✓（都是"测试本身测不到东西" ✗ 才暴露的 ✓）
1. 首版阴性对照写成"重建两条路径比数据顺序" ✗ —— 但 `_make()` 里会
   `torch.manual_seed(SEED)` **重新播种** ✓ ⇒ 两条路径 RNG 流**本就相同** ✗
   ⇒ 对照**测不到东西** ✗ ⇒ 改为**直接测机制** ✓（`capture → 抽 → restore → 抽` 必须重现 ✓）
2. 我按 `ckpt["model"]` 取值 ✗ —— 真实键是 **`delta`** ✗
   ⚠ 而 `ckpt.py` 注释明确"始终保存**完整** state_dict（不做 delta-only）" ✓
   ⇒ **键名有误导性** ✗（记录于此 ✓ 重命名属**行为变更** ✗ 应单独决策 ✓）
   真实顶层键 = `delta` / `opt` / `ema` / `scheduler` / `rng` / `rng_meta` / `train_steps` ✓

**守卫测试的定位** ✓: 除功能测试外，每个切片都配了**不变量守卫**（静态扫描 + 行为断言 ✓），
用于防止后续改动**回退**已修缺陷 ✗ —— 例如:
- `test_refactor_pr3.py`: 归档文件不得回到运行时路径 ✓；**活跃 CLI 入口不得被误归档** ✗
- `test_refactor_pr4d.py`: **注册顺序断言** ✓（防"整理"顺序 ✗）
- `test_refactor_pr4c.py`: `@torch.no_grad()` 存在性 ✓（锁死我踩过的坑 ✓）
- `test_refactor_pr1.py`: `np.empty` / `write_done` 必需产物 ✓

---

## 7. 进度对照审查 PR0–PR7

| PR | 内容 | 状态 |
|---|---|---|
| **PR0** | 黄金评测 / 训练 smoke / 身份记录 | ✅ 前向黄金 + smoke + manifest 齐备 ✓ **全程 bit-exact** ✓ |
| **PR1** | 修 P0 缺陷 + 统一 eval 完成协议 | ✅ **11 项全清** ✓（错图/缺图/错 STD **必报错** ✓）|
| **PR2** | LatentContract / RAM Dataset / REPA 缓存检查 | ✅ 完成 ✓（覆盖 100% 契约 ✓ 无静默 fallback ✓）|
| **PR3** | 单一采样器 / 评测 pipeline / 指标产物解耦 | 🟡 主体验收通过 ✓；**剩 1 项活跃重复采样** ✗（**需 GPU 逐位对比** ✓）|
| **PR4** | 拆训练主函数 / Loss / EMA / optimizer / checkpoint | 🟡 已 4 切片 ✓；余: **optimizer 组件** ✓ **CheckpointManager** ✓ **TrainStep/Trainer**（`main()` 仍 2429 行 ✓ 其中训练循环体 **937 行** ✗）|
| **PR5** | 拆 DiT 条件系统（Writer×Glyph 插件化）| ⏳ 未开始（依赖 PR4 ✓）|
| **PR6** | 单卡 profile / H2D / compile / DataLoader 优化 | ⏳ 未开始（**需 4090 空闲** ✓ 审查 §7.2 的 A–E 矩阵 ✓）|
| **PR7** | 归档旧代码 / 删重复入口 / 精简配置 | ⏳ 部分已做（2 个死 eval 已归档 ✓）|

---

## 8. 待办与阻塞（如实 ✓）

**唯一硬阻塞** ✓: 4090 正被 **v88 预训练**占用（另一会话 ✓ 只读观察 ✓ 未触碰 ✓）
⇒ 阻塞 **PR3 的 GPU 逐位对比** ✓ 与 **PR6 的 profiling** ✓。

**可继续（无需 GPU）** ✓:
1. PR4 余下组件: `optimizer`（L1500-1600 区 ✓ 含 weight-decay 排除逻辑 ⇒ 需谨慎 ✓）、`CheckpointManager` ✓
2. PR4 主体: `main()`（2429 行）→ `Trainer` + `TrainStep`（937 行循环体 ✓ 建议**分块**推进 ✓）
3. PR5: DiT 条件系统拆分（`ConditionBatch` 统一 ✓ 审查 §4.1 ✓）

**过程纪律（本会话教训 ✓ 已固化）**
- 共享仓库 ⇒ **只用显式路径** `git add` ✓（曾因 `git add -u src/eval` **误扫另一会话的在途改动** ✗ ⇒ 已 `--amend` 剥离并还原到工作区 ✓）
- 不使用 `ruff --fix src tests` ✗（会改动他人文件 ✓ 改为**逐文件**指定 ✓）
- 远程命令一律写成 `.sh` 文件 ✓（PowerShell 引号/中文会反复损坏 ✗）
- 大改动前先 **AST 结构测绘** ✓（本轮即用它选定切片 ✓）
- 每次提交前核对 `git diff` **仅含本次改动** ✓（工作区长期有他人 ~89 个改动文件 ✓）
