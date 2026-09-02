import os, sys
os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')
sys.stdout.reconfigure(encoding='utf-8')

from src.utils import MCCDLatentDataset

# A 段: base 训练数据 (img latent), 无图
print('=== A段 data (v8a) ===')
ds_a = MCCDLatentDataset(
    csv_file='5script/train_fame_clean_v8.csv', latent_shards_dir='final_latents_fame_v8',
    img_root=None, skel_root=None, skel_latent_shards_dir=None,
    image_size=256, load_canny=False, load_skel=False, is_train=True,
    preload=False, load_image=False, num_preload_workers=4, structure_size=256)
print('A samples:', len(ds_a))
s = ds_a[0]
print('A keys:', sorted(s.keys()))
print('A latent shape:', s['latent'].shape, s['latent'].dtype)

# B 段: ctrl 数据 (img latent + skel latent), REPA 关 => 无图
print('=== B段 data (v8b) ===')
ds_b = MCCDLatentDataset(
    csv_file='5script/train_fame_clean_v8.csv', latent_shards_dir='final_latents_fame_v8',
    img_root=None, skel_root='final_skel1_fame_v8',
    skel_latent_shards_dir='final_skel_latents_fame_1px_v8',
    image_size=256, load_canny=False, load_skel=False, is_train=True,
    preload=False, load_image=False, num_preload_workers=4, structure_size=256)
print('B samples:', len(ds_b))
s = ds_b[0]
print('B keys:', sorted(s.keys()))
print('B latent:', s['latent'].shape, '| skel_latent:', s['skel_latent'].shape, s['skel_latent'].dtype)

# B 段 REPA 开 => 需 image from final_imgs_fame_v8
ds_br = MCCDLatentDataset(
    csv_file='5script/train_fame_clean_v8.csv', latent_shards_dir='final_latents_fame_v8',
    img_root='final_imgs_fame_v8', skel_root='final_skel1_fame_v8',
    skel_latent_shards_dir='final_skel_latents_fame_1px_v8',
    image_size=256, load_canny=False, load_skel=False, is_train=True,
    preload=False, load_image=True, num_preload_workers=4, structure_size=256)
s = ds_br[0]
print('BR image shape:', s['image'].shape, s['image'].dtype, 'min/max:', float(s['image'].min()), float(s['image'].max()))

# eval cache 数据
print('=== eval v8 ===')
import csv
rows = list(csv.DictReader(open('5script/eval_fame_strict_clean_v8.csv', encoding='utf-8')))
print('eval rows:', len(rows), '| sample path:', rows[0]['image_path'])
print('ALL DATA SMOKE OK')