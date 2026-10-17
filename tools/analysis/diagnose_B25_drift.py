"""Paired pixel flips now; optional exact sampled-latent critic and 3-seed replay."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve
from src.eval.style.brush import features_from_image

B25 = 'assets/results/v86_full_fixed_B25/20261015-071058-v86-full-fixed-B25'
BASE = 'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond'
SOURCES = {'0k': BASE+'/eval_samples_ctrl/step0200000/eval200fix',
           '5k': B25+'/eval_samples_ctrl/step0005000/eval200fix',
           '10k': B25+'/eval_samples_ctrl/step0010000/eval200fix'}
CKPTS = {'0k': BASE+'/checkpoints/0200000.pt',
         '5k': B25+'/checkpoints/0005000.pt', '10k': B25+'/checkpoints/0010000.pt'}


def score(paths, enc, device, keys, prototypes, rows, class_keys):
    dataset = ImgSet(paths, tag='paired score')
    embeds = dataset.emb_all(enc, device)
    sims = embeds @ prototypes.T
    logits = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            logits.extend(enc(dataset.batch(list(range(i, min(i+32, len(rows)))), device)).cpu().tolist())
    result = []
    for i, row in enumerate(rows):
        target = row['calligrapher']+'·'+row['script']
        ci = keys.index(target)
        # Match eval_gen_with_disc: independent pixel margin includes the same-script STD rival.
        # Training latent margin intentionally excludes STD; keep those definitions distinct.
        rivals = [j for j, k in enumerate(keys) if j != ci and k.endswith('·'+row['script'])]
        pred = keys[int(sims[i].argmax())]
        feat = features_from_image(paths[i])
        result.append(dict(index=i, character=row['character'], writer=row['calligrapher'], script=row['script'],
            prototype_pred=pred, correct=pred == target, margin=float(sims[i, ci]-max(sims[i, rivals])),
            independent_target_logit=logits[i][class_keys.index((row['calligrapher'], row['script']))],
            independent_logits=logits[i], ink=feat.scalars['ink_frac'], widths=feat.width_stats()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--raw-critic', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    opt = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    rows = list(csv.DictReader(open(ROOT/'exp-std/csv/eval200_fixed.csv', encoding='utf-8-sig')))
    train = list(csv.DictReader(open(ROOT/'exp-std/csv/train.csv', encoding='utf-8-sig')))
    random.Random(0).shuffle(train)
    train = train[:6000]
    ck = torch.load(ROOT/'assets/style_disc_ce_holdout.pt', map_location='cpu', weights_only=False)
    enc = Enc(d=ck['d'], n_cls=ck['n_cls']).to(opt.device).eval()
    enc.load_state_dict(ck['model'])
    class_keys = [tuple(k) for k in ck['all_keys']]
    groups, paths = defaultdict(list), []
    for row in train:
        if row.get('aug', '').strip(): continue
        path = resolve(row['image_path'], [str(ROOT), str(ROOT/'data'), str(ROOT/'exp-std')])
        if path:
            groups[row['calligrapher']+'·'+row['script']].append(len(paths)); paths.append(path)
    emb = ImgSet(paths, tag='fixed bank6000').emb_all(enc, opt.device)
    protos = {k: emb[ix].mean(0) for k, ix in groups.items()}
    stdpaths, stdgroups = [], defaultdict(list)
    unique_std, seen_std = [], set()
    for row in rows:
        stdpaths.append(resolve(row['std_path'], [str(ROOT), str(ROOT/'data'), str(ROOT/'exp-std')]))
        key = (row['script'], row['character'])
        if key not in seen_std:
            seen_std.add(key)
            stdgroups['std·'+row['script']].append(len(unique_std))
            unique_std.append(stdpaths[-1])
    se = ImgSet(unique_std, tag='STD').emb_all(enc, opt.device)
    protos.update({k: se[ix].mean(0) for k, ix in stdgroups.items()})
    keys = sorted(protos)
    P = np.stack([protos[k]/np.linalg.norm(protos[k]) for k in keys])
    out = ROOT/('assets/B25_drift_raw_20261015' if opt.raw_critic else 'assets/B25_drift_cpu_20261015')
    out.mkdir(parents=True, exist_ok=True)
    per = {stage: score([str(ROOT/path/f'g{i}.png') for i in range(len(rows))], enc, opt.device,
                        keys, P, rows, class_keys) for stage, path in SOURCES.items()}
    regressions = [i for i in range(len(rows)) if per['5k'][i]['correct'] and not per['10k'][i]['correct']]
    improvements = [i for i in range(len(rows)) if not per['5k'][i]['correct'] and per['10k'][i]['correct']]
    for i in range(len(rows)):
        a = np.asarray(Image.open(ROOT/SOURCES['5k']/f'g{i}.png').convert('RGB'), dtype=np.float32)/255
        b = np.asarray(Image.open(ROOT/SOURCES['10k']/f'g{i}.png').convert('RGB'), dtype=np.float32)/255
        per['10k'][i]['pixel_l1_from_5k'] = float(np.abs(a-b).mean())
    summary = {stage: {'top1': sum(r['correct'] for r in records)/len(rows),
                       'margin': float(np.mean([r['margin'] for r in records]))} for stage, records in per.items()}
    result = dict(protocol='same-process independent pixel CE prototypes, 6000-ref bank',
                  device=opt.device, summary=summary, regressions=regressions, improvements=improvements,
                  per_image=per, independent_class_keys=class_keys,
                  paired_net_change=len(improvements)-len(regressions))
    discordant = len(regressions)+len(improvements)
    result['paired_exact_two_sided_p'] = min(1., 2*sum(math.comb(discordant, k)
        for k in range(min(len(regressions), len(improvements))+1))/2**discordant)
    result['regressions_with_ink_increase'] = sum(per['10k'][i]['ink'] > per['5k'][i]['ink'] for i in regressions)
    for stage, step in [('0k', 200000), ('5k', 5000), ('10k', 10000)]:
        run = BASE if stage == '0k' else B25
        native = list(csv.DictReader(open(ROOT/run/'eval_stdskel_summary.csv', encoding='utf-8-sig')))
        matching = [r for r in native if int(r['step']) == step and r.get('set') == 'eval200fix']
        if len(matching) != 1:
            raise ValueError(f'Expected one native metric row: {stage}: {matching}')
        summary[stage]['native_metrics'] = matching[0]
        summary[stage]['ink'] = float(np.mean([r['ink'] for r in per[stage]]))
    (out/'paired.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PAIRED', summary, 'correct_to_wrong=', regressions, 'wrong_to_correct=', improvements, flush=True)
    # Show every regression, not a handpicked set. Original PNGs are unmodified.
    fontpath = next((p for p in [Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
        if p.exists()), None)
    font = lambda size: ImageFont.truetype(str(fontpath), size) if fontpath else ImageFont.load_default()
    for page, start in enumerate(range(0, len(regressions), 6)):
        image = Image.new('RGB', (1540, 110+6*330), 'white'); draw = ImageDraw.Draw(image)
        draw.text((15, 10), 'B25 correct@5k -> wrong@10k | GT / STD / base / 5k / 10k', font=font(24), fill='black')
        draw.text((15, 42), 'head = independent pixel classifier, NOT latent Critic | M = same-script margin incl STD', font=font(18), fill='black')
        for j, i in enumerate(regressions[start:start+6]):
            y = 90+j*330
            label = f"#{i} {rows[i]['character']}" if fontpath else f'#{i}'
            draw.text((5, y), label, font=font(20), fill='black')
            pp = [resolve(rows[i]['image_path'], [str(ROOT), str(ROOT/'data'), str(ROOT/'exp-std')]),
                  stdpaths[i]]+[str(ROOT/SOURCES[s]/f'g{i}.png') for s in ('0k', '5k', '10k')]
            for col, path in enumerate(pp):
                image.paste(Image.open(path).convert('RGB').resize((256, 256)), (130+col*280, y))
                if col >= 2:
                    record = per[('0k', '5k', '10k')[col-2]][i]
                    draw.text((130+col*280, y+260), 'M=%.3f ink=%.3f\nhead=%.2f' %
                              (record['margin'], record['ink'], record['independent_target_logit']),
                              font=font(17), fill='black')
        image.save(out/f'regressions_{page+1:02d}.png')
    if not opt.raw_critic:
        return
    from src.eval.model_io import load_model_from_ckpt
    from src.eval.in_mem_eval import _get_cache
    from src.eval.sampler import sample_latents, build_diffusion
    from src.loss.critic_loss import CriticLoss
    from src.train.strong_style import real_mask, writer_margin
    from diffusers import AutoencoderKL
    critic = CriticLoss(str(ROOT/'assets/writer_brush_critic.pt'), opt.device)
    import lpips
    perceptual = lpips.LPIPS(net='vgg').to(opt.device).eval()
    vae = AutoencoderKL.from_pretrained(str(ROOT/'data/pretrained/sd-vae-ft-ema')).to(opt.device).eval()
    targets = torch.tensor([critic.slots.index(r['calligrapher']+'·'+r['script']) for r in rows], device=opt.device)
    sid = torch.tensor([critic.s2i[r['script']] for r in rows], device=opt.device)
    mask = real_mask(critic.slots, opt.device)
    raw_report = {}
    (out/'raw_protocol.json').write_text(json.dumps(dict(weights='EMA', sampler='heun', steps=50,
        cfg=1., sde_gamma=0., scaling_factor=.18215, shift=0., std_sigma=.65, seeds=opt.seeds,
        critic_slots=critic.slots, latent_margin='target - logsumexp(other real writers, same script)',
        pixel_margin='target prototype cosine - max(other same-script prototypes, including STD)',
        caveat='Seed0 is a regeneration: check per-image pixel L1 and aggregate replay fidelity before comparing with historical PNGs'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    for stage, checkpoint in CKPTS.items():
        model, args = load_model_from_ckpt(str(ROOT/checkpoint), device=opt.device, use_ema=True)
        args.std_soft_root = str(ROOT/'exp-std/data/std_soft_glyphs_v1')
        args.std_soft_eval_sigma = .65
        cache = _get_cache('exp-std/csv/eval200_fixed.csv', 187, None, '', args)
        g = cache['skels_latent']
        diff = build_diffusion(50, diffusion_type='flow', flow_kwargs={'sampler': 'heun', 'sde_gamma': 0.})
        raw_report[stage] = {}
        for seed in opt.seeds:
            noise = cache['noise'] if seed == 0 else torch.randn(cache['noise'].shape,
                generator=torch.Generator().manual_seed(seed), dtype=cache['noise'].dtype)
            z = sample_latents(model, diff, noise, cache['conds'], 1., 8, torch.device(opt.device),
                               skel=g, seed=seed, hier_conds=cache['hier_conds'])
            dest = out/f'{stage}_seed{seed}'; dest.mkdir(exist_ok=True)
            scores, brush, perceptual_scores = [], [], []
            for j in range(0, len(rows), 8):
                zs = z[j:j+8].to(opt.device)
                scored = critic.model(zs.float(), g[j:j+8].to(opt.device), sid[j:j+8])
                scores.append(scored['slot_logits'].cpu()); brush.append(scored['brush_logit'].cpu())
                with torch.autocast(opt.device, dtype=torch.bfloat16):
                    rgb = vae.decode(zs/.18215).sample.float().cpu().clamp(-1, 1)
                rgb = ((rgb+1)*127.5).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                quantized = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(opt.device).float()/127.5-1
                perceptual_scores.extend(perceptual(quantized, cache['gts'][j:j+8].to(opt.device)).flatten().cpu().tolist())
                for k, arr in enumerate(rgb): Image.fromarray(arr).save(dest/f'g{j+k}.png')
            logits = torch.cat(scores).to(opt.device)
            margins = writer_margin(logits, targets, mask)
            ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
            same_ce = torch.nn.functional.cross_entropy(logits.masked_fill(~mask[targets], -torch.inf), targets, reduction='none')
            records = score([str(dest/f'g{i}.png') for i in range(len(rows))], enc, opt.device,
                            keys, P, rows, class_keys)
            for i, record in enumerate(records):
                record.update(latent_writer_logits=logits[i].cpu().tolist(), latent_margin=float(margins[i]),
                              latent_ce26=float(ce[i]), latent_ce_same_script=float(same_ce[i]),
                              lpips_vgg=perceptual_scores[i])
                if seed == 0:
                    original = np.asarray(Image.open(ROOT/SOURCES[stage]/f'g{i}.png').convert('RGB'), dtype=np.float32)
                    regenerated = np.asarray(Image.open(dest/f'g{i}.png').convert('RGB'), dtype=np.float32)
                    record['regeneration_pixel_l1'] = float(np.abs(original-regenerated).mean()/255)
                    record['regeneration_hash_match'] = hashlib.sha256((dest/f'g{i}.png').read_bytes()).hexdigest() == hashlib.sha256((ROOT/SOURCES[stage]/f'g{i}.png').read_bytes()).hexdigest()
            raw_report[stage][str(seed)] = dict(top1=sum(r['correct'] for r in records)/len(rows),
                margin=float(np.mean([r['margin'] for r in records])), critic_ce26=float(ce.mean()),
                critic_ce_same_script=float(same_ce.mean()), critic_margin=float(margins.mean()), per_image=records)
            raw_report[stage][str(seed)]['lpips_vgg'] = float(np.mean(perceptual_scores))
            if seed == 0:
                replay_l1 = float(np.mean([r['regeneration_pixel_l1'] for r in records]))
                raw_report[stage][str(seed)]['replay_mean_pixel_l1'] = replay_l1
                raw_report[stage][str(seed)]['replay_fidelity_warning'] = replay_l1 > .01
            (out/'raw_critic_and_seeds.json').write_text(json.dumps(raw_report, ensure_ascii=False, indent=2), encoding='utf-8')
            print('RAW', stage, seed, raw_report[stage][str(seed)]['top1'], flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
