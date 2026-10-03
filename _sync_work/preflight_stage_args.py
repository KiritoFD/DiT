"""干跑: 用**真实 CLI 解析器**校验一段训练 argv —— 不建模型、不读数据、不训练。

为什么需要它 (2026-10-03 的两次真实事故都出在这一层):
  · `--fresh-scheduler` 是 type=_str_to_bool, 裸写 -> argparse 报
    "expected one argument" -> 阶段秒退 -> 级联中止 -> GPU 空转 7 小时;
  · ckpt 名零填充步数 `0017500.pt`, bash 算术按八进制 -> max_steps 算错。
两者都在"模块之外", 单测抓不到。此工具把 argv 喂给同一套 parser, 并断言
"传进去的 --max-steps / --early-stop-from-step 必须原样落地"。

用法 (级联运行器在每段启动前调用):
    python tools/preflight_stage_args.py -u src/train/train.py --config ... --max-steps 47500 ...
退出码: 0 通过; 非 0 = 不要启动该阶段。
"""
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

argv = list(sys.argv[1:])
# 允许运行器把完整命令(python -u src/train/train.py --xxx)原样递进来:
# 丢掉第一个 `--flag` 之前的所有 token (解释器选项 + 脚本路径)。
while argv and not argv[0].startswith("--"):
    argv.pop(0)
if not argv:
    print("[preflight] ✗ 没解析到任何 --flag, argv 传错了", file=sys.stderr)
    sys.exit(6)
sys.argv = ["train.py"] + argv
for _bad in ("--fresh-scheduler",):
    if _bad in argv:
        print(f"[preflight] ✗ argv 里出现 {_bad}: 本仓库它的语义是**步数计数器归零**, "
              f"会使 --max-steps 的绝对语义与 --early-stop-from-step 的绝对过滤失效。",
              file=sys.stderr)
        sys.exit(4)

from src.train.cli import parse_args  # noqa: E402

a = parse_args()          # argparse 的 unknown/缺值/类型错会在这里现形并 sys.exit


def _want(flag):
    return int(argv[argv.index(flag) + 1]) if flag in argv else None


w = _want("--max-steps")
if w is not None and int(a.max_steps) != w:
    print(f"[preflight] ✗ max_steps 被改写: 传 {w} 得 {a.max_steps}", file=sys.stderr)
    sys.exit(5)
w = _want("--early-stop-from-step")
if w is not None and int(getattr(a, "early_stop_from_step", -1)) != w:
    print(f"[preflight] ✗ early_stop_from_step 被改写: 传 {w} "
          f"得 {getattr(a, 'early_stop_from_step', None)}", file=sys.stderr)
    sys.exit(5)

print(f"[preflight] OK  config={os.path.basename(str(a.config))}  "
      f"max_steps={a.max_steps}  resume={getattr(a, 'resume_full', None)}  "
      f"resume_lr={getattr(a, 'resume_lr', None)}  "
      f"early_stop={getattr(a, 'early_stop', None)}"
      f"({getattr(a, 'early_stop_metric', None)})"
      f"  from_step={getattr(a, 'early_stop_from_step', None)}"
      f"  weights={getattr(a, 'skel_latent_shards_weights', None)}"
      f"  batch={getattr(a, 'global_batch_size', None)}"
      f"  compile={getattr(a, 'compile', None)}/{getattr(a, 'compile_mode', None)}"
      f"  in_mem_eval={getattr(a, 'in_mem_eval', None)}"
      f"({getattr(a, 'in_mem_eval_sets', None)})"
      f"  eval_mode={getattr(a, 'eval_mode', None)}"
      f"  w_repa={getattr(a, 'w_repa', None)}")
