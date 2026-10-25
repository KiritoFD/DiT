"""Heldout evaluation and acceptance for PixelWriterV2 encoders.

Evaluates:
1. Cross-character generalization (test_char).
2. Cross-source generalization (test_source).
3. Image-disjoint strictly non-overlapping 2AFC.
4. Perturbation robustness (blur, ink gain).
5. Background-only probe (erase ink strokes to verify no paper/scanner shortcut).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.writer_style_pixel import PixelWriterV2, WriterStylePixel, erase_ink_keep_background
from tools.train_writer_style_pixel import embed, evaluate_features, sha
from tools.analysis.run_style_fingerprint_audit import interval


def grouped_interval(features, pairs, data):
    margin = (features[pairs[:, 0]] * features[pairs[:, 1]]).sum(1) - (features[pairs[:, 0]] * features[pairs[:, 2]]).sum(1)
    correct = (margin > 0).astype(float) + 0.5 * (margin == 0)
    used = set()
    keep = []
    for i, tri in enumerate(pairs):
        if not used.intersection(tri):
            used.update(tri)
            keep.append(i)
    keep = np.array(keep, int)
    return {
        'by_char': interval(correct, data['char'][pairs[:, 0]]),
        'by_source': interval(correct, data['source'][pairs[:, 0]]),
        'image_disjoint': interval(correct[keep], data['char'][pairs[keep, 0]]),
        'mean_margin': float(margin.mean())
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--validation-only', action='store_true', help='Infrastructure smoke on val_char only')
    a = p.parse_args()
    torch.set_num_threads(4)
    run = Path(a.run)

    data = torch.load(Path(a.data_root) / 'data.pt', map_location='cpu', weights_only=False)
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)

    haar_mode = ck['config'].get('haar_mode', 'none')
    model = PixelWriterV2(n_slots=len(data['slots']), haar_mode=haar_mode).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)

    gray = data['gray']  # uint8 on CPU
    indices = np.arange(len(gray))
    tr = data['splits']['train']

    report = {
        'step': ck['step'],
        'checkpoint_sha256': sha(run / 'best.pt'),
        'data_manifest_sha256': ck['config']['data_manifest_sha256'],
        'arm': ck['config'].get('arm', 'custom'),
        'haar_mode': haar_mode,
        'metric_loss': ck['config'].get('metric_loss', 'ranking'),
        'conditions': 'image only; no STD/reference/writer/script input',
        'conditions_evaluated': {}
    }

    clean_features = None
    perturbations = ['clean', 'blur', 'ink_gain', 'background_only']

    for perturb in perturbations:
        if perturb == 'clean':
            features, pred = embed(model, gray, indices, perturb=None)
            clean_features = features
        elif perturb in ('blur', 'ink_gain'):
            features, pred = embed(model, gray, indices, perturb=perturb)
            features[tr] = clean_features[tr]  # Frozen clean gallery
        elif perturb == 'background_only':
            # Background Probe: erase strokes and replace with estimated background paper
            bg_features = []
            bg_pred = []
            for start in range(0, len(indices), 256):
                ids = indices[start:start + 256]
                raw = gray[torch.as_tensor(ids, dtype=torch.long)].cuda().float() / 255.0
                erased = erase_ink_keep_background(raw)
                with torch.no_grad():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = model(erased)
                bg_features.append(o['embedding'].float().cpu().numpy())
                bg_pred.extend(o['slot_logits'].argmax(1).cpu().tolist())
            features = np.concatenate(bg_features)
            features[tr] = clean_features[tr]  # Gallery remains clean
            pred = np.array(bg_pred)

        result = {}
        eval_keys = ('val_char',) if a.validation_only else ('test_char', 'test_source')
        for key in eval_keys:
            q = data['splits'][key]
            m = evaluate_features(features, pred, data, q, tr)
            m['matched'] = grouped_interval(features, data['matched_pairs'][key], data)

            # Cosine head margin
            weights = F.normalize(model.head.weight, dim=-1).detach().cpu().numpy()
            rng = np.random.default_rng(89)
            outcomes = []
            qidx = []
            for i in q:
                ns = q[(data['char'][q] == data['char'][i]) & (data['si'][q] == data['si'][i]) & (data['labels'][q] != data['labels'][i])]
                if len(ns):
                    j = int(rng.choice(ns))
                    margin = float((features[i] - features[j]) @ weights[data['labels'][i]])
                    outcomes.append(float(margin > 0) + 0.5 * float(margin == 0))
                    qidx.append(i)
            m['samechar_target_head_2afc'] = interval(outcomes, data['char'][qidx])
            result[key] = m

        result['train_slot_ce_top1'] = float((pred[tr] == data['labels'][tr]).mean())
        report['conditions_evaluated'][perturb] = result
        print(f"PIXEL_ACCEPTANCE {perturb}", json.dumps(result), flush=True)

    # Gradient probe in image space
    sample_ids = tr[:8]
    image = gray[torch.as_tensor(sample_ids, dtype=torch.long)].cuda().float() / 255.0
    image.requires_grad_(True)
    output = model(image)
    weights = F.normalize(model.head.weight, dim=-1)
    loss = -(output['embedding'] * weights[torch.from_numpy(data['labels'][sample_ids]).cuda()]).sum(1).mean()
    grad = torch.autograd.grad(loss, image)[0]

    report['gradient_probe'] = {
        'norm': float(grad.norm()),
        'finite': bool(torch.isfinite(grad).all()),
        'frozen_parameters': all(p.grad is None for p in model.parameters())
    }

    clean = report['conditions_evaluated']['clean']
    bg = report['conditions_evaluated']['background_only']

    # Acceptance criteria:
    # 1. Clean test_char matched accuracy >= 60%
    # 2. test_source 95% CI lower bound > 50%
    # 3. Background-only probe CE drops significantly towards chance level
    if a.validation_only:
        passgate = False
    else:
        cc = clean['test_char']['matched']['by_char']
        ss = clean['test_source']['matched']
        bg_ce = bg['train_slot_ce_top1']
        passgate = (cc['accuracy'] >= 0.60 and
                    min(ss['by_char']['ci95'][0], ss['by_source']['ci95'][0]) > 0.50 and
                    bg_ce < 0.20)

    report['research_gate_passed'] = bool(passgate)
    out_file = run / ('acceptance_smoke_val_only.json' if a.validation_only else 'acceptance.json')
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(run / 'clean_features.npz', embedding=clean_features)
    print('PIXEL_ACCEPTANCE_COMPLETE', run, flush=True)


if __name__ == '__main__':
    main()
