# 远程部署与运维

## 1. 环境拓扑

| 位置 | 角色 | 内容 |
|---|---|---|
| 开发机（Windows，`G:\GitHub\DiT`） | 代码编辑 / git | 全部代码 + 配置 + 文档；git 历史（branch main，remote KiritoFD/DiT） |
| 远程 4090 (`10.176.54.17:36430`, root, `/root/Workspace/xy/DiT`) | 训练 / GPU | **所有数据**（csv/图片/latent/ckpt），Python = `/opt/conda/bin/python`（有 numpy/torch/diffusers） |

**同步铁律：只同步代码文件到远程，数据全部留在远程**（数据目录不入 git，见 §5）。

SSH 别名（本地 `~/.ssh/config`）：

```
Host 4090
    HostName 10.176.54.17
    Port 36430
    User root
```

## 2. 网络与 SSH 注意事项（网络不稳）

- 网络反复超时（`Connection timed out` / `Connection closed` on 36430）。先探测再操作：
  ```powershell
  ssh 4090 "echo alive"
  ```
- **scp 与 ssh 分开调用**（scp 可能成功而 ssh 超时）。
- 每个 nohup 启动单独一次 ssh 调用。
- PowerShell 会破坏内联 `python -c`/awk/grep 的引号（含 `$`、逗号）→ **不要用内联 python -c**；先把 .py scp 上去再执行。
- 写文件的命令避免用 PowerShell `Get-Content/Set-Content`（会毁 UTF-8 中文）→ 用本仓库的 Python 脚本（io.open utf-8）或 scp 覆盖。

## 3. 常用运维命令

```bash
# 探测 / 看训练
ssh 4090 "echo alive"
ssh 4090 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
ssh 4090 "tail -20 /tmp/s19_midclean_train.log"
ssh 4090 "ls -lt /root/Workspace/xy/DiT/5script/results/s19_midclean_s_flow/*/checkpoints | head"

# 停训练（单卡）：
ssh 4090 "pkill -f 'train.py.*s19_midclean'"

# 起训练（nohup，先 cd 到项目目录，日志写 /tmp）：
ssh 4090 "cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python train.py --config s19_midclean_s_flow.json > /tmp/s19_midclean_train.log 2>&1 &"

# 起 CPU 指标 daemon（ControlNet eval 需要）：
ssh 4090 "cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python src/eval/eval_ctrl_metrics_daemon.py > /tmp/eval_ctrl_daemon.log 2>&1 &"
```

同步单个文件 / 目录：

```powershell
scp s19_midclean_s_flow.json 4090:/root/Workspace/xy/DiT/
scp -r src 4090:/root/Workspace/xy/DiT/
```

## 4. GPU 纪律（单卡 24G）

- 模型训练占 ~20.88G → **训练期间禁止任何并行 GPU 任务**（VAE 编码、GPU 批量评测都要等训练停）。
- 训练内 eval 复用训练进程自己的显存（换 EMA 权重 → 采样 → 恢复），不另开进程。
- 指标计算走独立 **CPU** daemon（LPIPS vgg 在 CPU 上跑）。

## 5. git 与目录纪律

- 本地 commit 是权威；远程 git 是旧 fork 历史，不直接回推。
- `.gitignore` 忽略数据目录：`5script/`、`results/`、`final_*`、`pretrained_models/`、`.venv/`、大日志与 png。
- 提交信息用「feat:/fix:/refactor:」前缀 + 中文说明（历史惯例）。
- `archive/diag_backup_20260828/` 放一次性诊断脚本（`_*.py`），不参与主流程与 py_compile。

## 6. 踩坑清单（血泪史）

| 坑 | 现象 | 解法 |
|---|---|---|
| flow 训练用 randint 取 t | loss 看似收敛但学到垃圾，skelIoU≈0.04 | 统一 `diffusion.sample_t()`（`02_diffusion.md` §4） |
| PowerShell 引号毁内联命令 | 远程执行报语法错 / 中文乱码 | scp .py 再执行；避免 python -c |
| Get-Content/Set-Content 毁 UTF-8 | eval 脚本 py_compile 报 U+E511/U+20AC | 从 git 恢复源文件；只用 Python 处理文本 |
| `load_main_model` 缺 `import os` | 重构后 NameError: name 'os' | src/model/controlnet.py 顶部补 import（已修） |
| pending marker 缺 `step_tag` | ctrl daemon 找不到评测目录 | `write_pending_metrics_marker` 必须写 `step_tag`（已修） |
| 根 launcher 调 `main()` 缺 args | TypeError | 统一走 `main_from_cli()`（已修） |
| `diffusion` shim 缺 `_extract_into_tensor` | import * 不带下划线符号 | shim 显式补一行（已修） |
| eval 目录重复污染 | 旧 eval_samples 与 round-robin | ckpt 轮换同步清理 + 时间戳隔离实验目录（已修） |

## 7. 例行检查（训练进行中）

1. `tail /tmp/s19_midclean_train.log`：loss 应单调下降；异常跳变先查是否 NaN/梯度。
2. `nvidia-smi`：显存稳定 ~20.9G，无其他进程占卡。
3. `ls {results_dir}/*/checkpoints`：按时出 ckpt（s19 每 2500 步）。
4. `eval_auto_*.json` 存在性与趋势（主模型 daemon 每 ckpt 出一次）。
5. 网络断了重连后：先 `echo alive` 确认 ssh，再继续下一步，不要盲目重启训练。