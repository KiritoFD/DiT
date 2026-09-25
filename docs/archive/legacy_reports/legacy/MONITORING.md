# 监控与自动化（MONITORING.md）

> 2026-08-14 ｜ 覆盖 `tools/pull_log.py` + `tools/train_dashboard.html` + `_watchdog.sh`（远程自动跑实验）。

---

## 1. Dashboard（训练曲线可视化）

**本地一条命令起服务**（在 `tools/` 目录）：

```bash
cd G:\GitHub\DiT\tools
python -m http.server 8731
```

浏览器打开 **http://127.0.0.1:8731/**（`index.html` 自动跳转 `train_dashboard.html`）。

**数据流**：
```
远程训练日志(exp_*.log) --scp--> tools/train_run.log --解析--> tools/train_data.json --fetch--> dashboard 图表
```

- `pull_log.py` 解析每 20 步的训练行（step/total/diff/canny/skel/repa/stepsPerSec/**Mem**）+ auto-eval 行（MSE/SSIM）。
- 图表：Diff / Total / 结构 loss / Steps-Sec / **显存（当前+峰值）** / MSE / SSIM。
- `eval_latest.png`（最新推理拼图）自动拉回并展示。

**拉取守护（每 60s）**：
```bash
python tools\pull_log.py --loop --interval 60
```

## 2. 远程自动跑实验（watchdog）

`_watchdog.sh` 在远程**脱离 SSH**（`setsid nohup`）按队列自动跑：`训练(tmux) → 轮询 Done! → eval_test 终评 → 下一个`。

```bash
# 远程启动
setsid nohup bash /root/Workspace/xy/DiT/_watchdog.sh > watchdog_out.log 2>&1 < /dev/null &
```

- **队列**（脚本内 `QUEUE` 数组）：`name|config|model|use_lora|lora_r|lora_target|pretrained|out`
- 每步 `sleep 30` 轮询；`Done!`=成功，`Error during training loop/CUDA out of memory/Traceback`=异常跳过。
- 结果追加写 `experiment_summary.txt`（`NAME: ckpt=... MSE=... SSIM=...`）。
- 异常不中断队列：`TRAIN_FAIL` / `EVAL_FAIL` / `NO_CKPT` 记录后继续。

**踩过的坑（务必遵守）**：
1. 队列里 config 名必须带 `.json` 后缀（否则 train.py 回落默认配置，5 秒空跑 Done）。
2. log 名 = `${config%.json}.log`（去后缀），否则找不到日志。
3. watchdog 自身的 stdout 重定向到 `watchdog_out.log`（不要指望 ssh 终端）。

## 3. 10k test 终评

`eval_test.py` 对任意 ckpt 在 10k test 上算 MSE/SSIM（单步重建，与训练 eval 同口径）：

```bash
/opt/conda/bin/python eval_test.py \
  --ckpt results/exp_xl_head/2026xxxx-DiT-3Cond-XL-2/checkpoints/0037000.pt \
  --model DiT-3Cond-XL/2 --use-lora 0 --num-calligraphers 1873 \
  --pretrained pretrained_models/DiT-XL-2-256x256.pt --out test_eval_B
```

输出 `test_eval_B.json`（MSE/SSIM）+ `test_eval_B_imgs/`（前 N 张对比图）。

## 4. 目录/日志约定

| 位置 | 内容 |
|---|---|
| `results/exp_<name>/<ts>-DiT-3Cond-XX-2/` | 实验 run（log.txt + checkpoints/ + eval_*） |
| `exp_<name>.log` | 根目录重定向的实时训练日志（watchdog 用） |
| `experiment_summary.txt` | watchdog 的最终结果汇总 |
| `tools/train_data.json` | dashboard 数据（git 忽略） |
