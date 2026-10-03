# BUG 档案: torch.compile 缓存打穿 -> 静默回退 eager -> 显存翻倍 OOM (2026-10-04)

## 症状 (表象极具欺骗性)

v52 (无骨架 + 书家表 + 字表, DiT-2Cond-Sp/2, batch 192, compile on) 在 **step ~251 OOM**:

- 稳态显存恒定 `14.44G/14.46G` (reserved/峰值), step 200 与 250 之间**没有任何爬升**;
- 然后**一步之内** allocated 冲到 22.03G, 报 `CUDA out of memory`;
- OOM 栈在 dit.py block/mlp 前向深处, 看起来像"batch 开太大"或"eval 显存尖峰"——都不是;
- 同 pattern 此前还出现过: batch192+骨架版在 step 1 backward OOM (22.19G), 当时也误判过。

## 真凶 (三层)

```
[rank0]:W torch._dynamo hit config.accumulated_cache_size_limit (256)
        function: 'torch_dynamo_resume_in_forward_at_2326' (src/model/dit.py:2326)
```

1. **数据依赖分支**: `LabelEmbedder.forward` (dit.py:128) 里的
   `if null_mask.any():` —— tensor 的 `.any()` 在 compiled 区域造成 graph break,
   且该布尔**逐批翻动** (cond_drop_all_prob=0.05, 56831 类的 CFG null 行),
   每次翻动 -> guard 失败 -> dynamo 重编译/新增缓存条目。
2. **缓存上限打穿**: ~250 步内 accumulated_cache_size_limit (默认 256) 被填满。
3. **静默回退 eager**: 达限后 torch.compile **不打日志直接回退 eager**;
   batch 192 下 eager 需要 ~22G (compiled 稳态 14.4G) -> 回退即 OOM。
   回退不产生任何警告行, 只有 OOM 栈里残留的
   `torch_dynamo_resume_in_forward_at_*` 帧名可以反推。

## 修复 (两处, 已上远程, 备份 *.bak_infra_*)

**A. `src/model/dit.py` LabelEmbedder.forward** —— 去掉数据依赖分支, 改无分支 where:

```python
# 旧: if null_mask.any(): out = torch.where(...)   <- graph break + 逐批重编译
null_mask = (labels == self.num_classes)
out = torch.where(null_mask.unsqueeze(-1), self.null_embed, out)   # 全 False = 恒等选择
```

语义完全等价 (全 False 时 where 是恒等映射), 代价一次 (B,D) elementwise select。
该类同时被 y_callig / y_char 两个 embedder 复用, 修复惠及所有 run。

**B. `src/train/train.py` compile 注入处** —— dynamo 上限防御:

```python
from torch import _dynamo as _td          # 勿写 import torch._dynamo: 函数内 import
_td.config.cache_size_limit = max(64, _td.config.cache_size_limit)   # 会把 torch 变成局部名
_td.config.accumulated_cache_size_limit = 4096                        # -> UnboundLocalError
model = torch.compile(model, mode=_compile_mode)
```

作用: 即使未来再出现重编译, 也是继续走编译路径, **不再静默掉回 eager**。

## 验证

batch 192 + compile on, 500 步 (旧故障点 251 的两倍):

```
rc=0, 500 步全跑完
Mem: 14.49G/14.51G  全程恒定 (无爬升/无回退)
无 accumulated_cache_size_limit 告警, 无 OOM
```

## 排查方法论 (下次直接复用)

- OOM 时先看**完整体栈**: `torch_dynamo_resume_in_forward_at_*` 帧名 = compiled 区域内炸的;
  紧邻 OOM 前的 dynamo W 级日志行是判决书;
- 日志的 `Mem: 活跃/峰值` 是 reserved 口径 (train.py:2384), 恒定不等于没有泄漏;
  **"allocated 突然多出 ~8G" 而不是线性爬升** = 单步事件 (回退/新图), 不是泄漏;
- 复现命令: `TORCH_LOGS=recompiles` + 300 步, 看 `Recompiling` 的 frame 与 guard 原因;
- `pkill -f 模式` 时若模式出现在 ssh 命令行里会把自己的 shell 一起杀 (踩过两次),
  把清理动作放进脚本文件、且模式不出现在调用命令行里。

## 相关文件

- 修复补丁: `_sync_work/probe_batch_v52.sh` [1] 段 (事务式、锚点校验、幂等)
- 诊断日志: `exp-std/logs_smoke/v52_recompile.log`
- 实验上下文: v52 = 无骨架 + 书家表(adaLN) + 字表(SupCon v2, 冻结),
  config `src/train/configs/v52_char_supcon_table.json`
