"""Phase 0: unified manifest, leakage and source audit -> data_audit.json.

Freezes the valid writer-script slots, character groups, source labels, and
the inventory of generation panels usable by eval_style_v2. Everything later
phases consume must appear here; nothing may be added informally afterwards.
"""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from .common import load_real_rows,save_json,slot_of,REAL_CSV,sha256_file


def protocol_csv_map(runs_root):
    """protocol name -> csv path, from the plan/protocols files beside the runs root."""
    mapping={}
    for up in range(4):
        base=Path(runs_root).parents[up]
        for name in ('plan.json','protocols.json'):
            f=base/name
            if not f.is_file():continue
            try:
                data=json.loads(f.read_text())
            except Exception:
                continue
            protocols=data.get('evaluations') or data.get('protocols') or []
            if isinstance(protocols,dict):protocols=list(protocols.values())
            for v in protocols:
                if isinstance(v,dict) and v.get('name') and v.get('csv'):
                    mapping.setdefault(v['name'],v['csv'])
    return mapping


def audit_panels(runs_root):
    """Discover completed evaluation panels (arm, step, protocol) with counts."""
    panels=[]
    root=Path(runs_root)
    if not root.exists():return panels
    protocol_map=protocol_csv_map(runs_root)
    for complete in sorted(root.glob('*/evaluations/*/*/complete.json')):
        try:
            identity=json.loads(complete.read_text())
        except Exception:
            continue
        folder=complete.parent
        pngs=list(folder.glob('eval_samples_ctrl/step*/eval200fix/*.png'))
        if not pngs:continue
        arm=complete.parents[3].name
        protocol=complete.parent.name
        csv_path=str(identity.get('csv') or identity.get('eval_csv') or protocol_map.get(protocol,''))
        panels.append(dict(arm=arm,step=int(complete.parents[1].name),protocol=protocol,
            csv=csv_path,
            checkpoint=str(identity.get('checkpoint') or ''),
            ema=bool(identity.get('ema',True)),n_png=len(pngs),
            png_root=str(folder)))
    return panels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs-roots',nargs='+',default=['assets/style_matrix_20261027/matrix64/48/execution/runs'])
    p.add_argument('--out',default='docs/1030/eval_style_v2/data_audit.json')
    a=p.parse_args()
    rows=load_real_rows()
    slot_rows=defaultdict(list)
    for r in rows:slot_rows[slot_of(r)].append(r)
    slots=[]
    for (w,s),rs in sorted(slot_rows.items()):
        chars={r['character'] for r in rs}
        sources=Counter(r.get('source','') for r in rs)
        slots.append(dict(writer=w,script=s,n=len(rs),n_chars=len(chars),
            sources=dict(sources),
            single_source=(len([v for v in sources.values() if v])<=1)))
    writers=sorted({s['writer'] for s in slots})
    scripts=sorted({s['script'] for s in slots})
    shared=None
    char_sets=[{r['character'] for r in rs} for rs in slot_rows.values()]
    if char_sets:shared=len(set.intersection(*char_sets))
    source_by_writer=defaultdict(set)
    for (w,s),rs in slot_rows.items():
        for r in rs:source_by_writer[w].add(r.get('source',''))
    collinear={w:sorted(v) for w,v in source_by_writer.items() if len([x for x in v if x])<=1}
    panels=[]
    for runs_root in a.runs_roots:panels+=audit_panels(runs_root)
    audit=dict(
        feature_protocol=dict(real_csv=REAL_CSV,real_csv_sha256=sha256_file(REAL_CSV),
            dedup='pixel_sha256',real_rows=len(rows)),
        n_writers=len(writers),n_scripts=len(scripts),n_valid_slots=len(slots),
        shared_chars_across_slots=shared,
        writer_source_collinear=collinear,
        slots=slots,
        panels=panels,
        leakage_rules=dict(
            writer_verification='references exclude the query character; competition same-script only; sources reported separately',
            nuisance_probe='any learned probe splits train/valid by character fold, never by image',
            distribution_metrics='references exclude eval-panel characters when the panel rows carry a character label'),
        missing=dict(ocr='no reliable Chinese OCR weights',ttf_control_bank='path to be frozen in phase 2'))
    save_json(a.out,audit)
    summary=dict(n_valid_slots=len(slots),shared_chars=shared,panels=len(panels),
        collinear_writers=sorted(collinear))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
