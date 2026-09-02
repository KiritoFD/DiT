import os, csv, glob, json, numpy as np

os.chdir('/root/Workspace/xy/DiT')

# --- 1) 谁在读 fame.npz? ---
print('=== who reads fame.npz ===')
for root, _, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding='utf-8').read()
            except Exception:
                continue
            if 'fame.npz' in txt:
                i = txt.find('fame.npz')
                print(' ', p, '|', ' '.join(txt[max(0, i - 80):i + 40].split()))
print()
for root, _, files in os.walk('_sync_work'):
    for f in files:
        if f.endswith('.sh'):
            p = os.path.join(root, f)
            txt = open(p, encoding='utf-8').read()
            if 'fame.npz' in txt:
                i = txt.find('fame.npz')
                print(' ', p, '|', ' '.join(txt[max(0, i - 60):i + 40].split()))

# --- 2) csv ids 在 final_imgs_fame_clean 的覆盖 ---
rows = list(csv.DictReader(open('5script/train_fame_clean.csv', encoding='utf-8')))
rows += list(csv.DictReader(open('5script/eval_fame_strict_clean.csv', encoding='utf-8')))
print()
print('train+eval csv rows:', len(rows))
ids = set()
missing = 0
dirs = {}
for r in rows:
    p = r['image_path']
    iid = int(p.split('/')[-1][:-4])
    ids.add(iid)
    d = p.split('/')[0]
    dirs[d] = dirs.get(d, 0) + 1
    if not os.path.exists('final_imgs_fame_clean/%d.png' % iid):
        missing += 1
print('unique ids:', len(ids), '| missing in final_imgs_fame_clean:', missing)
print('csv path dirs:', dirs)

# --- 3) 磁盘 ---
st = os.statvfs('.')
print()
print('disk free GB:', round(st.f_bavail * st.f_frsize / 2**30, 1))

# --- 4) final_imgs_fame_clean 文件数 ---
n = len(glob.glob('final_imgs_fame_clean/*.png'))
print('final_imgs_fame_clean pngs:', n)

# --- 5) fame.npz ids 是否 == csv ids? ---
npz = np.load('fame.npz')
npz_ids = set(npz['img_ids'].tolist())
print('fame.npz ids:', len(npz_ids), '| csv ids subset of npz:', ids.issubset(npz_ids), '| diff:', len(ids - npz_ids))