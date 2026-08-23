"""检查 bert-base-chinese 对我们数据集中所有字符的覆盖情况"""
import csv, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTHONIOENCODING'] = 'utf-8'
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained('bert-base-chinese')

# 读取所有唯一字符
chars = set()
with open('5script/train_top30_clean.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        chars.add(r['character'])
chars = sorted(chars)
print(f"总唯一字符: {len(chars)}")

# 检查每个字
known = []
unk = []
for c in chars:
    ids = t.encode(c, add_special_tokens=False)
    tokens = t.convert_ids_to_tokens(ids)
    if '[UNK]' in tokens or len(ids) != 1:
        unk.append(c)
    else:
        known.append(c)

print(f"BERT 词表中存在: {len(known)} ({len(known)/len(chars)*100:.1f}%)")
print(f"[UNK] 生僻字: {len(unk)} ({len(unk)/len(chars)*100:.1f}%)")
print(f"\nUNK 字符前20个: {''.join(unk[:20])}")
print(f"已知字符前20个: {''.join(known[:20])}")
