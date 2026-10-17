"""Gate the teacher, then prepare architecture-identical v84A continuation arms.

No reference/prototype condition, no writer swapping of low-noise GT states,
no table alignment, and no teacher parameters in generator checkpoints.
"""
import argparse
import copy
import json
import math
from pathlib import Path


def teacher_gate(matched,acceptance):
    results=matched['models']['v3']
    requirements={'test_char':(100,.40),'test_source':(50,.20)}
    failures=[]
    for name,(minimum,identity) in requirements.items():
        metric=results[name]
        if (metric['n']<minimum or metric['accuracy'] is None or not math.isfinite(metric['accuracy'])
                or metric['accuracy']<.60 or metric['ci95'] is None
                or not math.isfinite(metric['ci95'][0]) or metric['ci95'][0]<=.5):
            failures.append(name+': ink/width-matched cross-character discrimination insufficient')
        if acceptance['full_holdout_exported_prototypes'][name]['prototype_top1']<identity:
            failures.append(name+': writer identity generalization insufficient')
    return failures


def build_configs(base,teacher,bank,baseline):
    common=copy.deepcopy(base)
    common.update(max_steps=10000,epochs=10000,lr=1e-5,warmup_steps=300,
        ckpt_every=5000,epoch_steps=5000,global_batch_size=288,
        warm_start_from=str(baseline),resume_full=None,pretrained=None,
        writer_glyph_rank=0,style_ada_rank=0,strong_style_mode='off',
        w_critic_writer=0.,w_critic_away=0.,w_style_rank=0.,
        crosschar_gradient_ratio=0.,crosschar_encoder_ckpt='',crosschar_bank='',
        eval_dir='',num_workers=4,log_every=100,
        _comment='Architecture and inference identical to v84A. Train-only GT features used only in cross-character loss. No L_table or reference style conditioning.')
    control=copy.deepcopy(common)
    control.update(experiment_name='v89-cc-control',results_dir='assets/results/v89_crosschar_control')
    contrast=copy.deepcopy(common)
    contrast.update(experiment_name='v89-crosschar',results_dir='assets/results/v89_crosschar',
        crosschar_gradient_ratio=.20,crosschar_encoder_ckpt=str(teacher),crosschar_bank=str(bank),
        strong_style_probe_every=200,strong_style_max_weight=10.,strong_style_full_batch=True)
    return {'control':control,'crosschar':contrast}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-run',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    run=Path(a.teacher_run)
    matched=json.loads((run/'matched_teacher.json').read_text())
    acceptance=json.loads((run/'acceptance.json').read_text())
    failures=teacher_gate(matched,acceptance)
    output=Path(a.out);output.mkdir(parents=True,exist_ok=False)
    (output/'teacher_gate.json').write_text(json.dumps({'passed':not failures,'failures':failures,
        'limits':{'matched_accuracy':.60,'matched_ci_lower':.50,'min_char_pairs':100,'min_source_pairs':50,
                  'char_prototype_top1':.40,'source_prototype_top1':.20}},indent=2))
    if failures:
        raise SystemExit('TEACHER_NOT_ACCEPTED: '+'; '.join(failures))
    base=json.loads(Path('src/train/configs/v84A_std_cond.json').read_text())
    baseline=Path('assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt').resolve()
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    bank=output/'train_features.pt'
    configs=build_configs(base,(run/'best.pt').resolve(),bank.resolve(),baseline)
    for name,config in configs.items():
        (output/f'{name}.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print('V89_CROSSCHAR_PILOT_PREPARED',output,flush=True)


if __name__=='__main__':
    main()
