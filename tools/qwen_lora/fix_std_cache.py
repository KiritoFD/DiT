"""train_std.py: use precomputed text_std.pt; pipe without text encoder during training; eval lifts it."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')

old = """    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')"""
new = """    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    text_std = torch.load(root / 'text_std.pt', map_location='cpu', weights_only=True)
    pipe.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    print('TEXT_STD_LOADED', len(text_std), 'text encoder parked on CPU', flush=True)"""
assert s.count(old) == 1
s = s.replace(old, new)

old2 = """            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                per = [pipe.encode_prompt(prompt=[prompts[i]], image=[images[i]], device='cuda')
                       for i in range(config['batch'])]
                emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
                enc_mask = torch.nn.utils.rnn.pad_sequence(
                    [e[1][0] if e[1] is not None else torch.ones(e[0].shape[1], dtype=torch.bool,
                                                                 device=e[0].device) for e in per],
                    batch_first=True)
                img_pad = torch.nn.utils.rnn.pad_sequence(
                    [e[2][0] if e[2] is not None else torch.zeros(e[0].shape[1], dtype=torch.bool,
                                                                  device=e[0].device) for e in per],
                    batch_first=True)
                pixels"""
new2 = """            seqs = []
            real = []
            pads = []
            for row in chosen:
                e, pd = text_std[(row['character'], row['calligrapher'], row['script'])]
                seqs.append(e)
                real.append(torch.ones(e.shape[0], dtype=torch.bool))
                pads.append(pd if pd is not None else torch.zeros(e.shape[0], dtype=torch.bool))
            emb = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).cuda()
            enc_mask = torch.nn.utils.rnn.pad_sequence(real, batch_first=True).cuda()
            img_pad = torch.nn.utils.rnn.pad_sequence(pads, batch_first=True).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                pixels"""
assert s.count(old2) == 1, ('anchor2', s.count(old2))
s = s.replace(old2, new2)

old3 = """        out = root / 'evaluations' / f'{step:07d}'
        out.mkdir(parents=True, exist_ok=False)
        metrics = []
        model.eval()
        for row in val_rows:"""
new3 = """        out = root / 'evaluations' / f'{step:07d}'
        out.mkdir(parents=True, exist_ok=False)
        metrics = []
        model.eval()
        pipe.text_encoder.to('cuda'); torch.cuda.empty_cache()
        for row in val_rows:"""
assert s.count(old3) == 1
s = s.replace(old3, new3)

old4 = """        model.train()
        agg = {f: round(sum(m[f] for m in metrics) / len(metrics), 5)"""
new4 = """        model.train()
        pipe.text_encoder.to('cpu'); torch.cuda.empty_cache()
        agg = {f: round(sum(m[f] for m in metrics) / len(metrics), 5)"""
assert s.count(old4) == 1
s = s.replace(old4, new4)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('cache-based train_std patched')
