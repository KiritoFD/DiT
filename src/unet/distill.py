"""Writer-CFG Distillation for U-Net: distill CFG=1.6 velocity field into CFG=1.0."""
import argparse
import copy
from datetime import datetime
import json
import logging
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
import csv

from .data import loader, StdBank
from .io import Config, eval_args, diffusion
from .model import LatentUNet
from src.train.ema import update_ema
from src.train.step import build_cosine_scheduler
from src.eval.in_mem_eval import run_in_mem_eval

def parse_args():
    p = argparse.ArgumentParser(description="Writer-CFG Distillation")
    p.add_argument('--config', default='src/unet/configs/connectivity/DiC_U100_b768.json')
    p.add_argument('--teacher-ckpt', default='assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow/checkpoints/0050000.pt')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max-steps', type=int, default=30000)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--lambda-max', type=float, default=2.0)
    p.add_argument('--lambda-warmup', type=int, default=2000)
    p.add_argument('--teacher-cfg', type=float, default=1.6)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--ckpt-every', type=int, default=5000)
    p.add_argument('--output', default='assets/results/unet_writer_distill_30k')
    p.add_argument('--drop-writer-prob', type=float, default=0.08)
    p.add_argument('--is-control', action='store_true', help='Set lambda=0 for continuation control')
    return p.parse_args()

def main():
    cli = parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True

    conf_dict = json.loads(Path(cli.config).read_text(encoding='utf-8'))
    config = Config(**conf_dict)
    config.batch = cli.batch
    config.max_steps = cli.max_steps
    config.lr = cli.lr
    config.warmup = cli.warmup
    config.eval_every = cli.eval_every
    config.ckpt_every = cli.ckpt_every
    config.output = cli.output

    tag = "control" if cli.is_control else "distill"
    run_dir = Path(config.output) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-Unet-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'resolved_config.json').write_text(json.dumps(dict(cli=vars(cli), config=conf_dict), indent=2))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / 'train.log', encoding='utf-8')])
    log = logging.getLogger('distill')
    log.info("Starting Writer-CFG Distillation (is_control=%s, batch=%d, lr=%s, max_steps=%d)",
             cli.is_control, cli.batch, cli.lr, cli.max_steps)

    # 1. Load Teacher (Frozen DiC-768@50k EMA)
    log.info("Loading Teacher from %s...", cli.teacher_ckpt)
    teacher = LatentUNet(tuple(config.channels), config.num_writers, config.num_scripts,
                         num_res_blocks=2, residual_cond_gate=True,
                         stage_specific_cond=True, stage_cond_mode='adapter', act_fn='gelu').cuda()
    t_ck = torch.load(cli.teacher_ckpt, map_location="cpu", weights_only=False)
    t_weights = {k.removeprefix("_orig_mod."): v for k, v in t_ck["ema"].items()}
    teacher.load_state_dict(t_weights, strict=True)
    teacher.eval().requires_grad_(False)
    log.info("Teacher loaded and frozen.")

    # 2. Build Student (Initialized from exact same weights)
    student = LatentUNet(tuple(config.channels), config.num_writers, config.num_scripts,
                         num_res_blocks=2, residual_cond_gate=True,
                         stage_specific_cond=True, stage_cond_mode='adapter', act_fn='gelu').cuda()
    student.load_state_dict(t_weights, strict=True)
    student.drop_all = 0.0
    student.drop_writer = 0.0
    student.drop_script = 0.0
    student.train().requires_grad_(True)

    ema = copy.deepcopy(student).eval().requires_grad_(False)
    update_ema(ema, student, decay=1.0)
    log.info("Student initialized from Teacher weights; EMA initialized.")

    opt = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True)
    scheduler = build_cosine_scheduler(opt, warmup_steps=config.warmup,
                                       total_steps=config.max_steps, min_lr_ratio=config.min_lr_ratio)

    from .io import training_args
    args = training_args(config, run_dir)
    dataset, sampler, batches, tick, inline = loader(config, args, log, 0)
    bank = StdBank(config)
    diff = diffusion(config)

    step = 0
    running_fm = torch.zeros((), device='cuda')
    running_distill = torch.zeros((), device='cuda')
    running_total = torch.zeros((), device='cuda')
    log_start = perf_counter()

    batch_iter = iter(batches)

    while step < cli.max_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(batches)
            batch = next(batch_iter)

        step += 1
        x_latent = batch['latent'].to('cuda', non_blocking=True).float()
        y_callig = batch['y_callig_raw'].to('cuda', non_blocking=True)
        y_script = batch['y_script'].to('cuda', non_blocking=True)
        g_std = bank(batch, 'cuda', step)

        B = len(x_latent)
        t = diff.sample_t(B, x_latent.device) # in [0, 1)
        noise = torch.randn_like(x_latent)
        # Flow Matching linear interpolant
        x_t = (1.0 - t.view(B, 1, 1, 1)) * x_latent + t.view(B, 1, 1, 1) * noise
        v_fm = noise - x_latent

        # Coordinated Dropout: explicitly drop writer for 8% of samples
        drop_mask = (torch.rand(B, device='cuda') < cli.drop_writer_prob)
        y_callig_student = torch.where(drop_mask, torch.full_like(y_callig, config.num_writers), y_callig)

        # 3. Teacher Prediction (Frozen)
        if not cli.is_control:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Teacher conditional forward
                    v_w = teacher(x_t, t * 1000.0, g=g_std, y_callig=y_callig, y_script=y_script)
                    # Teacher unconditional forward (null writer)
                    v_null = teacher(x_t, t * 1000.0, g=g_std,
                                     y_callig=torch.full_like(y_callig, config.num_writers),
                                     y_script=y_script)
                    # v_style = v_null + cfg * (v_w - v_null) = v_w + (cfg - 1) * (v_w - v_null)
                    cfg_factor = cli.teacher_cfg - 1.0 # 0.6 for CFG=1.6
                    v_style = v_w + cfg_factor * (v_w - v_null)
                    # For dropped samples, target is v_null; for conditional samples, target is v_style
                    v_target = torch.where(drop_mask.view(B, 1, 1, 1), v_null, v_style)

        # 4. Student Forward
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_student = student(x_t, t * 1000.0, g=g_std, y_callig=y_callig_student, y_script=y_script)
            loss_fm = F.mse_loss(v_student.float(), v_fm.float())

            if cli.is_control:
                lambda_val = 0.0
                loss_distill = torch.zeros((), device='cuda')
                total_loss = loss_fm
            else:
                # Linear warmup 0 -> lambda_max over lambda_warmup steps
                lambda_val = min(cli.lambda_max, (step / max(1, cli.lambda_warmup)) * cli.lambda_max)
                loss_distill = F.mse_loss(v_student.float(), v_target.float())
                total_loss = loss_fm + lambda_val * loss_distill

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip, error_if_nonfinite=True)
        opt.step()
        scheduler.step()

        if step % config.ema_interval == 0:
            update_ema(ema, student, decay=config.ema_decay ** config.ema_interval)

        running_fm += loss_fm.detach()
        running_distill += loss_distill.detach()
        running_total += total_loss.detach()

        # Logging
        if step % config.log_every == 0:
            elapsed = perf_counter() - log_start
            n_steps = config.log_every
            cur_lr = opt.param_groups[0]['lr']
            log.info("Step %5d/%d | LR: %.2e | Total: %.4f | FM: %.4f | Distill: %.4f (λ=%.2f) | %.1f ms/step",
                     step, cli.max_steps, cur_lr,
                     running_total.item() / n_steps,
                     running_fm.item() / n_steps,
                     running_distill.item() / n_steps,
                     lambda_val,
                     (elapsed / n_steps) * 1000.0)
            running_fm.zero_()
            running_distill.zero_()
            running_total.zero_()
            log_start = perf_counter()

        # Periodic Checkpoint & Evaluation
        if step % cli.ckpt_every == 0 or step == cli.max_steps:
            ckpt_path = run_dir / 'checkpoints' / f"{step:07d}.pt"
            torch.save({
                'train_steps': step,
                'delta': student.state_dict(),
                'ema': ema.state_dict(),
                'opt': opt.state_dict(),
                'config': vars(config),
                'architecture': 'latent_cnn_unet_v1'
            }, ckpt_path)
            log.info("Saved checkpoint to %s", ckpt_path)

            # In-memory evaluation on eval200_fixed with EMA model at CFG=1.0 and CFG=1.6
            for eval_cfg in (1.0, 1.6):
                eval_args_cfg = eval_args(config)
                eval_args_cfg.eval_cfg = float(eval_cfg)
                eval_args_cfg.in_mem_eval_batch = 32
                eval_args_cfg.in_mem_eval_poster = False
                eval_sub_dir = run_dir / f"eval_step{step:07d}_cfg{eval_cfg:.1f}"
                eval_sub_dir.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    run_in_mem_eval(
                        ema, eval_args_cfg, step, 'cuda',
                        results_dir=str(eval_sub_dir)
                    )
                sum_p = eval_sub_dir / "eval_stdskel_summary.csv"
                if sum_p.exists():
                    r = list(csv.DictReader(open(sum_p)))[-1]
                    log.info("Eval Step %d (CFG=%.1f): SSIM=%.4f (med=%.4f), Frag=%.3f, Ink-IoU=%.4f, SoftInk=%.3f",
                             step, eval_cfg, float(r['ssim_mean']), float(r['ssim_med']),
                             float(r['frag_ratio']), float(r.get('ink_iou_mean', 0.0)), float(r['soft_ink_ratio']))

    log.info("Training complete: %d steps.", cli.max_steps)

if __name__ == '__main__':
    main()
