# remote.md — 远程机器调用规范

> 本文件是"如何安全、正确地操控远程机器"的操作手册。
> 远程机器：`ssh 4090`（不写主机/账号/端口等敏感信息）。
> 所有命令以 Windows PowerShell 为本地 shell。所有远程脚本以 bash 为远端 shell。

---

## 0. 三条铁律（违反必踩坑）

1. **绝不在 PowerShell 内联写多行 python/bash**（`python3 -c "...多行..."`、`ssh 4090 '...'` 内嵌 heredoc）。
   一律 **本地 `write` 成 `.py` / `.sh` 文件 → `scp` 到远程 → 远程执行**。
2. **远程文件保持 LF；绝不在远程跑 `sed -i "s/\r//g"`**。
   该命令经 Windows→ssh 传参时 `\r` 会变成 `r`，实际执行 `s/r//g` ——
   把文件里所有字母 `r` 删光（`root→oot`、`export→expot`），脚本瞬间报废。
   本地 write 工具产物本来就是 LF，无需任何转换。
3. **长任务挂 tmux，必须 `TERM=xterm` 前缀**。非交互 ssh 下 `TERM=dumb` 会导致 tmux server 起不来/挂死。

---

## 1. 基本调用

```powershell
# 单条命令（简单，无引号陷阱时可用）
ssh 4090 'pwd; ls'

# 挂长任务（训练/encode/eval），tmux 正确姿势：
ssh 4090 'TERM=xterm tmux new-session -d -s v8bc "bash /root/Workspace/xy/DiT/_sync_work/run_v8_bc.sh"'

# 查看会话
ssh 4090 'tmux ls'

# 之后所有 ssh 连接都不需要再带 TERM（server 已常驻）
```

> tmux 会话名用短名（`v8bc`、`enc`、`evald`），避免默认日期乱码。
> 不用 tmux 的替代：`setsid nohup bash script.sh > /tmp/x.log 2>&1 < /dev/null &`。

---

## 2. 引号与转义（PowerShell → ssh 双层解析）

| 场景 | 正确写法 | 反面教材 |
|---|---|---|
| 简单命令 | `ssh 4090 'tail -5 /tmp/x.log'` | `"$(...)"` 会被 PowerShell 本地展开 |
| 命令内用双引号 | `ssh 4090 'grep "step=" /tmp/x.log \| tail -3'` | 内层双引号用 `\"` 会坏 |
| `$(...)` / 变量 | **避免**；必须时用 `'bash -c "..."'` 或写成脚本 scp | PowerShell 先吞 `$()` |
| 管道 | `ssh 4090 'ls \| grep x'`（`\|` 转义）或写成脚本 | `|` 在 PowerShell 侧被解释为主机管道 |

**判断标准：先在本地 PowerShell 里把单引号内的字符串原样打印一遍，确认无 PowerShell 展开，再发给 ssh。**

---

## 3. 文件传输（唯一可靠路径）

```powershell
# 本地 → 远程
scp G:\GitHub\DiT\_ot_scratch\run_v8_bc.sh 4090:/root/Workspace/xy/DiT/_sync_work/run_v8_bc.sh

# 远程 → 本地
scp 4090:/root/Workspace/xy/DiT/_ot_scratch/v8_dash/evals_summary.csv G:\GitHub\DiT\_ot_scratch\v8_dash\
```

**scp 后不要 sed、不要任何字符级处理**（本地 write 是 UTF-8/LF，直传即用）。

---

## 4. 路径策略（不依赖时间戳）

**教训**：训练器在 `results_dir` 下按时间戳建子目录（`v8b/20260902-234912-.../checkpoints/`），
runner 用 `ls -dt dir_*` 猜最新目录，glob 少一层就断链。

**规范**：
- 所有跨阶段引用（A→B→C 的 ckpt）**先 copy 到固定无时间戳路径**再向下传：
  - `5script/results/v8_3stage/A_main_final.pt`（A best）
  - `5script/results/v8_3stage/B_ctrl_best.pt`（B best）
- 脚本内直接写死固定路径，**禁止 `ls -dt` + glob 猜目录**。
- 多级目录查找最多只出现在"选出 best"这一步（eval json 扫 ssim），选完立刻 copy 成固定名。

---

## 5. 环境速查（远程）

| 环境 | 用途 | 注意 |
|---|---|---|
| `/opt/conda/envs/cu121/bin/python` | **训练**（py3.10.18, torch2.1.2+cu121, xformers, compile 缓存） | 唯一的训练环境 |
| `/opt/conda/bin/python` | CPU eval daemon / 数据脚本 / 选 ckpt | 轻脚本用 |
| `TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor` | compile 持久化缓存 | 每个训练脚本必须 export |
| `PYTHONPATH=/root/Workspace/xy/DiT` | 训练脚本 import | 每个训练脚本必须 export |

**base 环境（py3.10.8, torch1.13）是 Golden Rule，永不升级。**

---

## 6. 训练链路约定（三段 v8 链）

```
A(train.py v8a base) 早停 ──▶ B(train_controlnet.py v8b, --main-ckpt A_main_final.pt)
                         B 早停(best ctrl ssim) ──▶ C(train_repa.py v8c, --main-ckpt + --ctrl-ckpt B_ctrl_best.pt)
```

- 每段参数**全量固化进 config json**，CLI 只传 ckpt 路径。
- 每段退出码/日志：`/tmp/v8b_s31_ctrl.log`、`/tmp/v8c_s32_repa.log`；链日志 `/tmp/v8_3stage.log`。
- **每段开始前**：确认 GPU 空闲 `nvidia-smi`，确认 tmux 会话在 `tmux ls`。
- **每段结束后**：把选优 ckpt copy 到固定路径，立即写文档 + git commit。

---

## 7. 监控约定

- 训练中按需求周期（默认 1200s）查：`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`
   + `tail -3 <段日志>` + `tail -3 /tmp/v8_3stage.log`。
- eval 产物：`<结果目录>/checkpoints/eval_auto_*.json`（A 段）或 `eval_auto_ctrl_*.json`（B 段）。
- 从不在训练机上跑与训练争 GPU/CPU 的基准（先杀基准进程再测）。

---

## 8. 常见坑速查

| 症状 | 根因 | 修复 |
|---|---|---|
| `cd: /oot/Wokspace/...: No such file` | 远程 sed `s/r//g` 删了 r | 重传干净文件，不再 sed |
| tmux "no server running" / `lost server` | TERM=dumb / pane 命令秒退 | `TERM=xterm tmux new-session -d ...` |
| `python -c` 报 syntax error | PowerShell 内联多行坏了 | write 脚本 + scp |
| ssh 命令超时挂死 | 命令含 `$(...)` 或内层引号被吞 | 拆成单引号简单命令或写脚本 |
| 时间戳目录 glob 选错/选空 | `ls -dt` 猜目录少一层 | 固定路径 copy，不用 glob |