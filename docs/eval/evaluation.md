# Evaluation Pipeline

This document covers the DiT calligraphy-generation evaluation pipeline: the two-process
training/evaluation split, the CPU auto-evaluation daemon
([`auto_eval_cpu.py`](../../auto_eval_cpu.py)), the in-memory evaluation core
([`eval_auto.py`](../../eval_auto.py)), the MSE/SSIM metrics, early stopping, and the
monitoring dashboards.

The pipeline exists because evaluating diffusion checkpoints is slow and CPU-bound, while
training is GPU-bound. Running both on the same process would either stall training or
underutilize the CPU. The design below keeps the two concerns fully decoupled while still
letting training read back evaluation results for early stopping.

---

## Evaluation Architecture

The evaluation system is designed to be **non-blocking**: training runs on GPU while
evaluation runs on CPU as an independent process. The two communicate only through the
filesystem — there is no shared memory, socket, or IPC queue.

### Two-Process Design

1. **`train.py` (GPU)** — Trains the model and saves checkpoints. Each checkpoint is written
   as a `.pt` file plus a `.done` marker, and `train.py` also writes a single
   `_active_ckpt_dir.txt` file at the experiment-results root pointing to the *active*
   checkpoint directory for the current run.
2. **`auto_eval_cpu.py` (CPU)** — Polls the active checkpoint directory. When it finds a new
   `.pt` whose `.done` marker is present (i.e. the checkpoint is fully written), it loads
   the checkpoint, rebuilds the model + VAE on CPU, runs evaluation, and writes
   `eval_auto_{step:07d}.json` with the MSE/SSIM results.

### Communication Between Processes

All communication is via files on disk:

| Writer | Path | Purpose |
|--------|------|---------|
| `train.py` | `{results_dir}/_active_ckpt_dir.txt` | Path to the active experiment's checkpoint dir (`train.py:145`) |
| `train.py` | `{ckpt_dir}/{step:07d}.pt` | The checkpoint itself (`delta`/`ema`/`opt`/`args`/`train_steps`) |
| `train.py` | `{ckpt_dir}/{step:07d}.pt.done` | Marker confirming the checkpoint is fully saved (`train.py:1061`) |
| `auto_eval_cpu.py` | `{ckpt_dir}/eval_auto_{step:07d}.json` | Per-checkpoint eval result: `{step, mse, ssim}` (`auto_eval_cpu.py:413`) |
| `auto_eval_cpu.py` | `{ckpt_dir}/cpu_eval_state.json` | Which checkpoints have been evaluated, plus errors/timestamps (`auto_eval_cpu.py:441`) |
| `auto_eval_cpu.py` | `{ckpt_dir}/eval_latest.png` | Latest `show5` preview sheet (overwritten each round) |
| `auto_eval_cpu.py` | `{ckpt_dir}/eval_samples/stepXXXXXXX/` | Per-step sample PNGs (history, not overwritten) |
| `auto_eval_cpu.py` | `{ckpt_dir}/seen_samples/stepXXXXXXX/` | Per-step training-set sample PNGs |

`train.py` reads back `eval_auto_*.json` to make early-stopping decisions — that is the
*only* direction training depends on evaluation, and it is a soft dependency (a missing
JSON simply means early stopping is not triggered yet).

### Why `.done` Markers?

`torch.save` is not atomic. If `auto_eval_cpu.py` polled for `*.pt` files directly, it
could load a half-written checkpoint. The `.done` marker is created *after* `torch.save`
returns (`train.py:1061`), so the eval daemon treats `.pt` + `.pt.done` as the "checkpoint
ready" signal (`auto_eval_cpu.py:527`).

### Experiment Switching

Each `train.py` launch creates a fresh timestamped experiment directory
(`train.py:141`) and overwrites `_active_ckpt_dir.txt`. The eval daemon detects the path
change and fully resets: it shuts down the worker pool, drops the cached model/VAE/caches,
rebuilds them for the new experiment, and loads a fresh `cpu_eval_state.json`
(`auto_eval_cpu.py:504`).

---

## `auto_eval_cpu.py`

The standalone CPU evaluation daemon. It runs forever (unless `--once`), polling the active
checkpoint directory at `--interval` seconds and evaluating any new `.done` checkpoints.

### Launch

```bash
tmux new-session -d -s evalcpu \
  'python auto_eval_cpu.py \
    --results-dir 5script/results/s7_klf4_top30 \
    --workers 8 --worker-threads 8 \
    --seen5-csv 5script/seen5_top30.csv'
```

Key flags (see `auto_eval_cpu.py:447` for the full parser):

| Flag | Default | Meaning |
|------|---------|---------|
| `--results-dir` | — | Training results dir; read `_active_ckpt_dir.txt` from here |
| `--ckpt-dir` | — | Pin a specific ckpt dir directly (overrides polling) |
| `--interval` | 30 | Poll interval in seconds |
| `--once` | off | Process all currently-ready checkpoints once and exit |
| `--device` | `cpu` | `cpu` or `cuda` (cuda only for a dedicated GPU eval box) |
| `--threads` | 0 | torch threads for the parent (0 = all cores; forced to 1 when `--workers>1`) |
| `--workers` | 1 | Number of persistent fork workers for `eval100` data parallelism |
| `--worker-threads` | 8 | torch threads *per worker* |
| `--seen5-csv` | `5script/seen5_top30.csv` | Training-set CSV for the `seen5` preview |
| `--eval-n` / `--steps` / `--cfg` / `--batch` | from ckpt args | Override eval params without touching training |

### How It Works (per poll iteration)

The main loop is `auto_eval_cpu.py:494` (`main()`). Each iteration:

1. **Resolve the active ckpt dir** by reading `_active_ckpt_dir.txt`
   (`read_active_ckpt_dir`, `auto_eval_cpu.py:423`). If the path has changed since the last
   iteration, reset model/VAE/caches/pool and reload `cpu_eval_state.json`.
2. **Scan for `.pt` files** and skip any already recorded in `state` (`auto_eval_cpu.py:523`).
3. **Skip checkpoints without `.done`** — they are still being written
   (`auto_eval_cpu.py:527`).
4. **Load the checkpoint** with `torch.load(..., weights_only=False)` and extract its `args`
   (model config) and `train_steps` (`auto_eval_cpu.py:531`).
5. **Build model/VAE on first run or when args change** (`build_model`, `load_vae`,
   `build_caches` at `auto_eval_cpu.py:557`). On subsequent checkpoints of the same
   experiment, only the weights are reloaded — the model architecture and caches are reused.
6. **Run `eval_one`** (`auto_eval_cpu.py:338`), which performs the three evaluations below
   and writes `eval_auto_{step:07d}.json`.
7. **Persist state** to `cpu_eval_state.json` so a restart never re-evaluates a checkpoint
   (`auto_eval_cpu.py:589`).

### The Three Per-Checkpoint Evaluations

`eval_one` (`auto_eval_cpu.py:338`) runs, in order:

1. **`eval100`** — Free-sampling DDIM metrics. `n=100` samples from the eval CSV, DDIM with
   50 steps and `cfg=4.0`, decoded through the VAE and compared to GT by MSE/SSIM. This is
   the number that drives early stopping. It is the slowest stage, so it is dispatched to
   the persistent worker pool *first* and the parent runs the visualizations in parallel.
2. **`show5`** — Fixed 5 unseen samples → `eval_latest.png` (overwritten) and
   `eval_samples/stepXXXXXXX/` (history) (`auto_eval_cpu.py:366`).
3. **`seen5`** — 5 training-set samples → `seen_samples/stepXXXXXXX/`
   (`auto_eval_cpu.py:379`). Pure visualization, never enters any metric.

### Weight Loading Priority

When loading a checkpoint, EMA weights win when available, falling back to the training
weights otherwise (`load_ckpt_weights`, `auto_eval_cpu.py:114`):

```python
sd = ckpt.get("ema")        # EMA weights (preferred when use_ema=true)
if sd is None:
    sd = ckpt.get("delta", ckpt.get("model"))  # Training weights fallback
```

`ckpt["delta"]` is produced by `train.py`'s `extract_full_inference` in LoRA mode (the
LoRA deltas + condition heads + adaLN/final_layer); in full-pretrain mode (`use_lora=false`)
it is the complete `state_dict()` (`train.py:1041`). The `ema` key is only present when
EMA is enabled (`train.py:1056`).

### VAE-Aware Parameters (from ckpt args)

The model's latent geometry is determined entirely by the VAE, and those parameters are
read from the checkpoint's `args` so the eval process never has to guess. Defaults assume
the legacy `sd-vae-ft-ema` f8 VAE (`_vae_params`, `auto_eval_cpu.py:136`):

| Parameter | f8 VAE | f4 VAE | Use |
|-----------|--------|--------|-----|
| `latent_channels` | 4 | 3 | Noise shape `z = randn(n, latent_channels, S, S)` |
| `latent_spatial` | `image_size // vae_downscale` = 32 (256/8) | 64 (256/4) | Spatial size of the latent |
| `scaling_factor` | 0.18215 | 0.102079 | Divide latent by this before `vae.decode` |

The eval cfg dict is built at `auto_eval_cpu.py:543` and threads these values through to
`eval_gen_in_memory` so the same code path works for both f4 and f8 checkpoints.

### Multiprocess Eval Pool

The `eval100` stage is the bottleneck. To use all CPU cores, `auto_eval_cpu.py` forks a
pool of **persistent workers** at startup and reuses them for every checkpoint
(`start_pool`, `auto_eval_cpu.py:237`).

- **`--workers 8`** forks 8 persistent workers at startup. Each worker is assigned a fixed
  slice of the `eval100` samples (contiguous range, balanced by count,
  `auto_eval_cpu.py:241`).
- Workers inherit the already-loaded model, VAE, and cache via `fork` (copy-on-write), so
  memory overhead per worker is minimal — they share the read-only tensors.
- On each eval round, the parent posts the checkpoint path to every worker
  (`pool_submit`, `auto_eval_cpu.py:270`); each worker `torch.load`s the weights into its
  inherited model, runs `eval_gen_in_memory` over its slice, and returns `(mse, ssim)`.
- The parent aggregates results with a count-weighted average (`pool_collect`,
  `auto_eval_cpu.py:285`).
- **`WORKER_TIMEOUT = 1800`** seconds (30 min) caps a single eval round
  (`auto_eval_cpu.py:193`). Any worker failure, timeout, or death marks the pool `broken`
  and the round falls back to single-process evaluation; the broken pool is never reused.
- While the workers run `eval100`, the parent process runs `show5` and `seen5`
  visualizations in parallel (`auto_eval_cpu.py:366`–`389`).

### Key Design Notes (the fork-before-threading rule)

This design was hard-won. Two previous bugs are documented in the source comments
(`auto_eval_cpu.py:176`–`189`):

1. **step2000 hung for 7 hours.** The fork happened *after* the parent had already run torch
   multithreaded work, so the OpenMP thread pool had been initialized. Child processes
   inherited a corrupted/dead thread pool and hung.
2. **step3000 was slow (40 min vs the expected 9 min).** Even after forcing
   `set_num_threads(1)` before a per-round fork, the already-built OpenMP pool could not be
   shrunk back down, and workers stayed slow.

**Root-cause fix:** fork the worker pool **exactly once, at startup, before any torch
multithreading has run** (`auto_eval_cpu.py:237`, called from `main` at
`auto_eval_cpu.py:575` while the parent is still single-threaded — see the
`torch.set_num_threads(1)` guard at `auto_eval_cpu.py:476`). After that one fork, the parent
is free to use multithreading for its own visualization work, because it never forks again.
Workers reuse the same pool for the lifetime of the experiment; on experiment switch the
pool is shut down and a fresh one is forked (`auto_eval_cpu.py:506`, again from a
single-threaded parent).

---

## `eval_auto.py`

The in-memory evaluation core. Two flavors of evaluation live here:

- `eval_in_memory` (`eval_auto.py:134`) — single-step `x_start` reconstruction at a fixed
  `t = T_EVAL = 150`. This is the older "reconstruction" eval; it measures how well the
  model predicts the clean latent from a partially noised GT latent, not true generation.
- `eval_gen_in_memory` (`eval_auto.py:221`) — **free-sampling** DDIM from pure noise. This
  is the one used by `auto_eval_cpu.py` for `eval100`/`show5`/`seen5`; it matches inference
  and measures actual generation quality.

The module docstring notes the historical motivation: the single-step reconstruction eval
"only tests reconstruction and misleads generation quality," so the free-sampling path
replaced it for in-training auto-eval (`eval_auto.py:193`–`195`).

### `eval_gen_in_memory(model, vae, device, cache, ...)`

The core free-sampling evaluation function (`eval_auto.py:221`). It generates images from
noise conditioned only on `(calligrapher_id, character_id)` — the GT image is never fed to
the model, only used as the comparison target.

For each batch in `cache`:

1. **Build the condition** from `cache["conds"]` — a list of
   `(calligrapher_id, script_id, character_id)` tuples (`eval_auto.py:258`). In `2cond`
   mode the script id is dropped.
2. **Sample initial noise** `z = torch.randn(j - i, 4, 32, 32, device=device)`
   (`eval_auto.py:247`). If `glyph_init_mix < 1.0` and a standard-glyph latent `g` is
   available, the starting point is mixed: `z = α·noise + (1-α)·g`
   (`eval_auto.py:250`–`255`) — this is the HYBRID init described in `HYBRID_INIT_PLAN.md`.
3. **DDIM denoise** (default 50 steps) via `ddim.ddim_sample_loop(model.forward_with_cfg,
   z.shape, z, model_kwargs=mk, ...)` (`eval_auto.py:275`). Inside `forward_with_cfg` the
   model runs both the conditional and unconditional forward passes and combines them with
   classifier-free guidance: `eps = uncond + cfg_scale * (cond - uncond)`.
4. **Decode** the denoised latent back to pixel space:
   `img = vae.decode(samples / scaling_factor).sample` (`eval_auto.py:277`). The
   `scaling_factor` (0.18215 for f8, 0.102079 for f4) undoes the latent normalization the
   VAE was trained with.
5. **Compare with GT**: accumulate MSE over the batch and per-image SSIM
   (`eval_auto.py:279`–`281`).

Returns `(mse, ssim)` as count-weighted averages over `n` (`eval_auto.py:311`).

> Note: `eval_auto.py` hardcodes `4`/`32`/`0.18215` for the noise shape and decode scaling
> in its own body (`eval_auto.py:247`, `:277`); the VAE-aware `latent_channels` /
> `latent_spatial` / `scaling_factor` keyword args exist on the signature for the
> f4-compatible call path used by `auto_eval_cpu.py`. When porting to a new VAE, pass the
> correct values from the checkpoint's `args` (as `auto_eval_cpu.py` does) rather than
> relying on the defaults.

### `prepare_gen_cache(dataset, n, cond_mode)`

Pre-loads `n` samples from a dataset into an in-memory cache (`eval_auto.py:197`) so the
diffusion loop never touches the dataloader at eval time:

- `conds`: list of `(calligrapher_id, [script_id,] character_id)` tuples
  (`eval_auto.py:205`–`209`). The middle element is `-1` in `2cond` mode.
- `gts`: list of GT images as `(3, 256, 256)` tensors in `[-1, 1]` (`eval_auto.py:210`,
  `:216`).
- `gs`: optional standard-glyph latents, used for HYBRID init when the dataset returns a
  `g` field (`eval_auto.py:212`).

The cache is built once per experiment by `build_caches` (`auto_eval_cpu.py:151`) and
shared with workers via fork.

### Visualization Helpers

- `_save_eval_visuals` (`eval_auto.py:18`) — stacks the first `vis_n` `(pred | GT)` pairs
  vertically into a single PNG (the `eval_latest.png` dashboard preview).
- `_dump_eval_all` (`eval_auto.py:53`) — writes every `(pred | GT)` pair to individual
  files under a directory (used when `vis_out` is not a `.png`/`.jpg`).
- The per-step `eval_samples/stepXXXXXXX/` directory is written by
  `eval_gen_in_memory` itself (`eval_auto.py:290`–`308`): each sample is saved as
  `sample{i}.png` + `gt{i}.png`, plus a `samples.json` metadata file with the step, cfg,
  steps, seed, and condition list.

---

## Metrics

| Metric | Description | Range | Good |
|--------|-------------|-------|------|
| MSE | Pixel MSE between generated and GT (256×256, `[-1,1]` range) | 0–4 | < 0.1 |
| SSIM | Structural similarity (11×11 Gaussian window, per-channel for multi-ch) | 0–1 | > 0.9 |

### MSE

Plain per-pixel mean squared error between the generated and GT images, both in `[-1, 1]`
(`eval_auto.py:279`). Because the data range is 2.0, the theoretical maximum MSE is 4.0
(opposite-extreme pixels); a typical "good" calligraphy generation sits below 0.1.

### SSIM Implementation

Standard Wang et al. SSIM (`_ssim`, `eval_auto.py:85`):

- **Single-channel**: 11×11 Gaussian window (σ=1.5) built by `_gaussian_window`
  (`eval_auto.py:78`). The window is applied via `F.conv2d` to compute local means,
  variances, and the cross-covariance, then combined into the SSIM map
  (`eval_auto.py:91`–`97`).
- **Multi-channel (RGB)**: per-channel recursion — `_ssim` calls itself once per channel and
  averages (`eval_auto.py:86`–`88`).
- **Data range**: the SSIM constants use `data_range=1.0` because the images are rescaled to
  `[0, 1]` (`(x + 1) / 2`) before the call (`eval_auto.py:281`); the underlying `[-1, 1]`
  range therefore maps to `data_range=2.0`, but the rescaled call uses 1.0.

SSIM is computed **per image** (not batch-pooled) with a Python loop
(`eval_auto.py:280`–`281`) — the comment at `eval_auto.py:165` notes that batch pooling
would mix images and corrupt the metric.

### VAE Floor Noise (reconstruction, no diffusion)

The VAE is not a lossless codec — encoding then decoding a GT image introduces a small
reconstruction error even with no diffusion model in the loop. This "floor" sets a lower
bound on achievable eval MSE:

| VAE | Floor MSE | Floor SSIM |
|-----|-----------|------------|
| `sd-vae-ft-ema` (f8) | 0.0037 | 0.9655 |
| `kl-f4` (f4) | 0.0019 | 0.9882 |

Eval MSE = VAE floor + diffusion error. The floor is < 1% of a typical eval MSE, so the
*bulk* of the measured error comes from the diffusion model, not the codec — which is the
desired property for a generation-quality metric.

---

## Historical Eval Results

### sd-vae f8 (previous training)

| Experiment | Dataset | Final Step | Eval MSE | Eval SSIM |
|-------------|---------|------------|----------|-----------|
| s6 top6 diff-only | top6 (10k) | 195k | 0.432 | 0.732 |
| s5 top30 diff-only | top30 (129k) | 70k | 0.841 | 0.520 |

### kl-f4 (current s7 training)

- Training started 2026-08-23.
- First eval at step 5000.
- Expected: lower MSE ceiling due to the 2× lower VAE floor noise (0.0019 vs 0.0037).

---

## Early Stopping

Early stopping lets training halt automatically once the eval metric stops improving. It is
driven entirely by the `eval_auto_*.json` files written by `auto_eval_cpu.py` — there is no
in-process evaluation during training.

### Mechanism in `train.py`

The check runs inside the training loop, gated by a step-multiple check so it does not fire
every step (`train.py:1086`):

```python
if (getattr(args, 'early_stop', False)
        and train_steps >= int(getattr(args, 'early_stop_min_steps', 0))
        and args.ckpt_every > 0
        and train_steps % _es_check_every == 0
        and rank == 0
        and _early_stop_check()):
    early_stop_stopped = True
    break  # stop training
```

`_early_stop_check()` (`train.py:600`) does the bookkeeping:

1. Glob the checkpoint dir for `eval_auto_*.json` and pick the highest step
   (`train.py:605`–`609`).
2. Skip if that eval step has already been processed (`train.py:610`) — prevents double
   counting when `_es_check_every` is smaller than the checkpoint interval.
3. Read the metric (`ssim` or `mse`) from the JSON (`train.py:616`–`620`).
4. Update `early_stop_best` / `early_stop_stale`. If the new value beats the best, reset
   stale to 0; otherwise increment it (`train.py:625`–`630`).
5. If `stale >= patience`, return `True` to stop (`train.py:634`–`637`).

The "better" direction depends on the metric: `ssim` is higher-is-better, `mse` is
lower-is-better (`train.py:595`).

### `_es_check_every`

How often (in training steps) the early-stop check runs. Default
`max(ckpt_every // 2, 1000)` — i.e. twice per checkpoint cycle, but never more often than
every 1000 steps (`train.py:596`–`598`). Checking more frequently than checkpoints is fine
because `_early_stop_check` dedupes by eval step.

### Current Config

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--early-stop-metric` | `ssim` | Maximize SSIM |
| `--early-stop-patience` | 6 | Stop after 6 consecutive non-improving evals |
| `--early-stop-min-steps` | 60000 | Do not stop before 60k steps |
| `--early-stop-check-every` | 0 → `ckpt_every // 2` = 2500 | Check every 2500 steps |

With `ckpt_every = 5000`, six consecutive non-improvements ≈ 30k steps of no SSIM
improvement before training stops.

---

## Dashboard

Two helper scripts in [`tools/`](../../tools/) turn the on-disk logs and eval JSONs into
static HTML dashboards:

- **[`tools/pull_monitor.py`](../../tools/pull_monitor.py)** — Pulls training logs
  (`log.txt`) and `eval_auto_*.json` files from the remote training box to a local mirror.
- **[`tools/build_dashboards.py`](../../tools/build_dashboards.py)** — Generates static HTML
  dashboards from the pulled logs.
- **[`tools/dashboards/index.html`](../../tools/dashboards/index.html)** — Dashboard entry
  point.

The dashboards render: loss curves, learning rate, EMA decay, steps/sec throughput,
MSE/SSIM over time, and the sample preview images (`eval_latest.png` and per-step
`eval_samples/`).
