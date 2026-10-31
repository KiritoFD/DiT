"""Print the fields of a D-arm spec that decide what actually trains."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'assets/d_20261101/specs/D1.json'
s = json.load(open(path, encoding='utf-8'))
c = s['config']
keys = ('lr', 'lr_backbone', 'lr_style_repr', 'lr_style_inj', 'lr_writer_table',
        'warmup', 'lr_decay_steps', 'ckpt_every', 'eval_every', 'batch', 'microbatch',
        'data_csv', 'ema_decay')
print('arm            :', s.get('arm'))
print('kind           :', s.get('kind'))
print('freeze_backbone:', s.get('freeze_backbone'))
print('skip_fm        :', s.get('skip_fm'))
print('decoder_lr     :', s.get('decoder_lr'))
print('joint          :', s.get('joint'))
print('milestones     :', s.get('milestones'))
print('source         :', s.get('source'))
for k in keys:
    print(f'{k:<15}:', c.get(k))
