with open('exp.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update status
text = text.replace(
    '**在训**：已达 ~49,000 步，速度 ~2.75 step/s，显存 13.80G。45k 完整 checkpoint/评测已落盘通过。',
    '**在训**：已达 ~50,300 步（50.3% 过半），速度 ~2.76 step/s，显存 13.80G。50k 完整 checkpoint/评测已落盘通过。'
)

# Search for 45k line and append 50k line
lines = text.splitlines(keepends=True)
new_lines = []
for line in lines:
    new_lines.append(line)
    if '| **45k** | 23.04M |' in line:
        new_lines.append('| **50k** | 25.60M | **0.59926** | **0.33352** | **0.32017** | **2.5848** | `eval_auto_50000.json`、`0050000.pt.done`（50%进度，[50k生成海报](docs/experiments/ledger/unet_50000_poster.png)已归档） |\n')

with open('exp.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('exp.md updated successfully with 50k')
