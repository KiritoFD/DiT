"""Decision rules for the F plan: K selection, arm selection, and the gates.

All rules are explicit and auditable — no hidden weighting. Ordering follows the
user's acceptance priority: content must not regress first, then writer
direction, then writer amplitude, then image quality.
"""
import argparse
import json
from pathlib import Path

# Content is "not regressed" if it stays within these deltas of the reference.
SSIM_TOL = 0.010
INK_IOU_TOL = 0.010
FRAG_REL_TOL = 0.15      # frag may grow by 15% relative before we call it worse


def load(path):
    return json.loads(Path(path).read_text())


def content_key(ev):
    c = ev['content']
    return dict(ssim=c.get('ssim'), ink_iou=c.get('ink_iou'),
                frag=c.get('frag'), lpips=c.get('lpips'),
                skel_iou=c.get('skel_iou'))


def writer_key(ev):
    w = ev.get('writer') or {}
    return dict(direction=w.get('direction'), amplitude=w.get('amplitude'),
                ocr_top1=(w.get('ocr') or {}).get('top1'))


def content_ok(cand, ref):
    """cand does not regress content relative to ref."""
    c, r = content_key(cand), content_key(ref)
    why = []
    if c['ssim'] is not None and r['ssim'] is not None and c['ssim'] < r['ssim'] - SSIM_TOL:
        why.append(f"ssim {c['ssim']:.4f} < {r['ssim']:.4f}-{SSIM_TOL}")
    if c['ink_iou'] is not None and r['ink_iou'] is not None and c['ink_iou'] < r['ink_iou'] - INK_IOU_TOL:
        why.append(f"ink_iou {c['ink_iou']:.4f} < {r['ink_iou']:.4f}-{INK_IOU_TOL}")
    if c['frag'] is not None and r['frag'] is not None and c['frag'] > r['frag'] * (1 + FRAG_REL_TOL):
        why.append(f"frag {c['frag']:.3f} > {r['frag']:.3f}*{1 + FRAG_REL_TOL}")
    return (not why), why


def score(ev):
    """Higher is better: direction dominates, amplitude only breaks near-ties."""
    w = writer_key(ev)
    d = w['direction'] or 0.0
    a = w['amplitude'] or 0.0
    return d + 0.10 * min(a, 1.5)


def cmd_pick_k(a):
    """K = the checkpoint with the better content that does not lose writer."""
    cands = {p.stem: load(p) for p in map(Path, a.evals)}
    names = sorted(cands)
    if len(names) < 2:
        raise SystemExit('pick-k needs at least two eval.json files')
    best = max(names, key=lambda n: score(cands[n]))
    # prefer the higher-content one unless its writer score is clearly worse
    ranked = sorted(names, key=lambda n: (content_key(cands[n])['ssim'] or 0), reverse=True)
    top = ranked[0]
    winner = top if score(cands[top]) >= score(cands[best]) - 0.05 else best
    out = dict(stage='pick_k', winner=winner,
               table={n: dict(content=content_key(cands[n]), writer=writer_key(cands[n]),
                              score=round(score(cands[n]), 4)) for n in names},
               rule='higher ssim unless writer score is >0.05 below the best')
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_pick_arm(a):
    """F0 vs F1: F1 must beat F0 on writer direction without losing content."""
    f0, f1 = load(a.f0), load(a.f1)
    ok1, why1 = content_ok(f1, f0)
    d0, d1 = (writer_key(f0)['direction'] or 0), (writer_key(f1)['direction'] or 0)
    beats = d1 > d0 + a.min_gain
    winner = 'F1' if (ok1 and beats) else 'F0'
    out = dict(stage='pick_arm', winner=winner,
               f0=dict(content=content_key(f0), writer=writer_key(f0)),
               f1=dict(content=content_key(f1), writer=writer_key(f1)),
               f1_content_ok=ok1, f1_content_why=why1,
               direction_gain=round(d1 - d0, 4), min_gain=a.min_gain,
               rule='F1 wins only if content holds AND direction beats F0 by min_gain')
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('pick-k'); k.add_argument('--evals', nargs='+', required=True)
    k.add_argument('--out', required=True); k.set_defaults(fn=cmd_pick_k)
    m = sub.add_parser('pick-arm'); m.add_argument('--f0', required=True)
    m.add_argument('--f1', required=True); m.add_argument('--out', required=True)
    m.add_argument('--min-gain', type=float, default=0.03)
    m.set_defaults(fn=cmd_pick_arm)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
