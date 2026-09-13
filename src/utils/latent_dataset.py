import os
import re
import glob
import csv
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from .callig_map import map_callig_id as _map_callig


def _load_one(task):
    """Module-level worker for preload: read one PNG as uint8 array."""
    i, path, mode, *shape = task
    with Image.open(path) as im:
        array = np.asarray(im.convert(mode), dtype=np.uint8)
    if shape:
        size = int(shape[0])
        if size == 32:
            if array.shape != (256, 256):
                raise ValueError(f"expected 256x256, got {array.shape}: {path}")
            array = array.reshape(size, 8, size, 8).max(axis=(1, 3))
    return i, array


class MCCDLatentDataset(Dataset):
    """
    Latent-cached + image dataset:
      - latent   : 从预构建 shard 加载（跳过 on-the-fly VAE encode）
      - image    : 256x256 原始图（仅 REPA 等需要 GT 图时加载）
      - canny    : 256x256 canny 图（canny loss）
      - skeleton : 256x256 skeleton 图（skel loss）

    preload=True 时在启动阶段把所需数据一次性读入内存（并行 PNG 解码），
    训练过程中零磁盘 IO —— 适合内存充足的大内存机器。
    csv rows image_path 形如 `final_images/<img_id>.png` 或 `data/imgs/final_imgs_256/<img_id>.png`。
    """
    def __init__(self, csv_file, latent_shards_dir, img_root,
                 image_size=256, is_train=False, preload=False, load_image=True,
                 num_preload_workers=16, use_glyph_cond=False, skel_latent_shards_dir=None,
                 callig_id_map=None, aux_latent_shards_dirs=None):
        self.samples = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                self.samples.append(row)
        # 书家词表收紧: 稀疏 raw calligrapher_id -> 连续索引 (见 callig_map.py)
        self._callig_map = callig_id_map
        self.use_glyph_cond = bool(use_glyph_cond)
        if self.use_glyph_cond:
            # 标准字形 latent 查询(懒加载, 全局单例), 训练/推理一致
            #
            # 注意：这里必须用 **v2** 字典。历史 bug：原本接的是 v1
            # (get_glyph_lookup -> std_glyph_latent)，而该目录在远程并不存在，
            # 导致命中率 **0%**；又因下方缺失时返回零张量，整个 w_glyph_cond
            # 条件是静默失效的 —— 不报错、loss 正常下降，但条件从头到尾是 0。
            # v2 (std_glyph_latent_v2) 才是真实存在的库（楷/行/隶 6 字体）。
            from src.utils import get_glyph_lookup_v2
            self._glookup = get_glyph_lookup_v2()
            self._report_glyph_coverage()
        else:
            self._glookup = None
        self._g_hit = 0
        self._g_miss = 0

        self.latent_shards_dir = latent_shards_dir
        self.img_root = img_root
        self.skel_latent_shards_dir = skel_latent_shards_dir
        self.image_size = image_size
        self.load_image = load_image

        self._shard_cache = {}
        self._id_to_shard = {}
        shards = sorted(glob.glob(os.path.join(latent_shards_dir, "shard_*.npz")))
        if not shards:
            raise FileNotFoundError(f"No shards in {latent_shards_dir}")
        # Auto-detect latent shape from first shard (supports f8=4ch/32x32, f4=3ch/64x64, etc.)
        _probe = np.load(shards[0])
        _lat = _probe["latents"]
        self.latent_channels = int(_lat.shape[1])
        self.latent_spatial = int(_lat.shape[2])
        _probe.close()
        for sp in shards:
            d = np.load(sp)
            for j, iid in enumerate(d["img_ids"]):
                self._id_to_shard[int(iid)] = (sp, j)
            d.close()

        # skel 条件: 优先 VAE latent shards (ControlNet latent 条件), 否则 PNG
        self._skel_id_to_shard = {}
        if self.skel_latent_shards_dir:
            _sk_shards = sorted(glob.glob(
                os.path.join(self.skel_latent_shards_dir, "shard_*.npz")))
            if not _sk_shards:
                raise FileNotFoundError(
                    f"No skel latent shards in {self.skel_latent_shards_dir}")
            _probe2 = np.load(_sk_shards[0])
            self.skel_latent_channels = int(_probe2["latents"].shape[1])
            self.skel_latent_spatial = int(_probe2["latents"].shape[2])
            _probe2.close()
            for sp in _sk_shards:
                d = np.load(sp)
                for j, iid in enumerate(d["img_ids"]):
                    self._skel_id_to_shard[int(iid)] = (sp, j)
                d.close()

        self.is_train = is_train
        self.preload = preload
        # ── aux latent 通道 (moyi 式辅助任务: 与图像 latent 一起作为扩散目标) ──
        # 每个 dir 一个 (N,4,32,32) shard 集 (keyed by img_id); 训练目标 x = cat(image, *aux)
        self.aux_latent_shards_dirs = list(aux_latent_shards_dirs or [])
        self.aux_meta = []            # [(channels, spatial), ...]
        self._aux_id_to_shard = []    # list[dict[id] -> (shard_path, j)]
        self.aux_channels = 0
        for _adir in self.aux_latent_shards_dirs:
            _sh = sorted(glob.glob(os.path.join(_adir, "shard_*.npz")))
            if not _sh:
                raise FileNotFoundError(f"No aux latent shards in {_adir}")
            _pr = np.load(_sh[0])
            self.aux_meta.append((int(_pr["latents"].shape[1]),
                                  int(_pr["latents"].shape[2])))
            self.aux_channels += int(_pr["latents"].shape[1])
            _pr.close()
            _mp = {}
            for _sp in _sh:
                _d = np.load(_sp)
                for _j, _iid in enumerate(_d["img_ids"]):
                    _mp[int(_iid)] = (_sp, _j)
                _d.close()
            self._aux_id_to_shard.append(_mp)
        self._latents = None
        self._imgs = None
        self._skel_latents = None
        self._aux_latents = None
        if preload:
            self._preload_all(num_preload_workers)

    def __len__(self):
        return len(self.samples)

    def _get_latent(self, img_id):
        """Load one latent for the non-preload path.

        The shard index is built in ``__init__``. Keep this path intentionally
        stateless so DataLoader workers do not retain many decompressed npz shards.
        """
        try:
            shard_path, offset = self._id_to_shard[int(img_id)]
        except KeyError as exc:
            raise KeyError(f"latent not found for img_id={img_id}") from exc
        with np.load(shard_path) as shard:
            latent = np.array(shard["latents"][offset], copy=True)
        return torch.from_numpy(latent)

    # ------------------------------------------------------------------ preload
    def _preload_all(self, num_workers=16):
        import multiprocessing as mp
        from collections import defaultdict
        t0 = time.time()
        n = len(self.samples)
        ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in self.samples]

        # --- latents: group by shard, load each shard once, scatter into RAM ---
        self._latents = np.empty((n, self.latent_channels, self.latent_spatial, self.latent_spatial), dtype=np.float32)
        by_shard = defaultdict(list)  # shard_path -> [(csv_idx, j)]
        for i, iid in enumerate(ids):
            sp, j = self._id_to_shard[iid]
            by_shard[sp].append((i, j))
        for sp, items in by_shard.items():
            d = np.load(sp)  # whole shard into RAM (npz, non-mmap)
            lat = d["latents"]
            for i, j in items:
                self._latents[i] = lat[j]
            d.close()
        print(f"[preload] latents {n:,} loaded in {time.time() - t0:.1f}s "
              f"({self._latents.nbytes / 1024 ** 3:.1f}G)")

        def _pool_fill(tasks, out, desc):
            done = 0
            with mp.Pool(num_workers) as pool:
                for i, a in pool.imap_unordered(_load_one, tasks, chunksize=512):
                    out[i] = a
                    done += 1
                    if done % 50000 == 0:
                        print(f"[preload] {desc} {done:,}/{n:,} "
                              f"({time.time() - t0:.0f}s)")
            print(f"[preload] {desc} {n:,} loaded in {time.time() - t0:.1f}s "
                  f"({out.nbytes / 1024 ** 3:.1f}G)")

        if self.load_image:
            self._imgs = np.empty((n, 256, 256, 3), dtype=np.uint8)
            tasks = []
            for i, r in enumerate(self.samples):
                p = r["image_path"]
                full = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
                if not os.path.isfile(full) and self.img_root:
                    full = os.path.join(self.img_root, f"{ids[i]}.png")
                tasks.append((i, full, "RGB"))
            _pool_fill(tasks, self._imgs, "images")

        # --- skel latent shards (latent 条件) ---
        if self._skel_id_to_shard:
            self._skel_latents = np.empty(
                (n, self.skel_latent_channels, self.skel_latent_spatial,
                 self.skel_latent_spatial), dtype=np.float32)
            by_sk = defaultdict(list)
            for i, iid in enumerate(ids):
                sp, j = self._skel_id_to_shard[iid]
                by_sk[sp].append((i, j))
            for sp, items in by_sk.items():
                d = np.load(sp)
                lat = d["latents"]
                for i, j in items:
                    self._skel_latents[i] = lat[j]
                d.close()
            print(f"[preload] skel latents {n:,} loaded in {time.time() - t0:.1f}s "
                  f"({self._skel_latents.nbytes / 1024 ** 3:.1f}G)")

        # --- aux latents (moyi 式辅助目标通道) ---
        if self._aux_id_to_shard:
            self._aux_latents = []
            for _k, _mp in enumerate(self._aux_id_to_shard):
                _ch, _sp_ = self.aux_meta[_k]
                _arr = np.empty((n, _ch, _sp_, _sp_), dtype=np.float32)
                _by = defaultdict(list)
                for i, iid in enumerate(ids):
                    _s, _j = _mp[iid]
                    _by[_s].append((i, _j))
                for _s, _items in _by.items():
                    _d = np.load(_s)
                    _lat = _d["latents"]
                    for i, _j in _items:
                        _arr[i] = _lat[_j]
                    _d.close()
                self._aux_latents.append(_arr)
                print(f"[preload] aux[{_k}] {n:,} loaded ({_arr.nbytes / 1024 ** 3:.1f}G)")

        total = (self._latents.nbytes
                 + (self._imgs.nbytes if self._imgs is not None else 0)
                 + (self._skel_latents.nbytes if self._skel_latents is not None else 0)
                 + (sum(a.nbytes for a in self._aux_latents)
                    if self._aux_latents is not None else 0))
        print(f"[preload] ALL preloaded in {time.time() - t0:.1f}s, "
              f"total RAM {total / 1024 ** 3:.1f}G")

    # --------------------------------------------------- glyph cond 自检
    def _report_glyph_coverage(self, log_every=0):
        """启动前统计标准字形条件的覆盖率并显式打印。

        存在的意义：历史 bug 中 v1 字典目录不存在、命中率 0%，但缺失被静默降级
        成零张量，导致 w_glyph_cond 实验全程无效却毫无征兆。这里强制在启动时
        把覆盖率打出来，并在命中率为 0 时升级为醒目告警，使这类失效无法再隐身。
        """
        if self._glookup is None:
            return
        total = hit = 0
        per_script = {}
        for row in self.samples:
            sid = int(row["script_id"])
            ch = row.get("character", "")
            sc = row.get("script", str(sid))
            total += 1
            ok = bool(ch) and self._glookup.get(sid, ch) is not None
            hit += int(ok)
            d = per_script.setdefault(sc, [0, 0])
            d[0] += 1
            d[1] += int(ok)
        rate = hit / max(total, 1)
        lines = [f"[glyph-cond] coverage {hit}/{total} = {rate*100:.1f}%"]
        for sc, (n, h) in sorted(per_script.items(), key=lambda x: -x[1][0]):
            lines.append(f"    {sc:<6} {h:>6}/{n:<6} {h/max(n,1)*100:>5.1f}%")
        if rate == 0.0:
            lines.insert(0, "=" * 58)
            lines.insert(1, "  !! FATAL: 标准字形条件命中率为 0% —— g 将全部为零张量，")
            lines.insert(2, "     w_glyph_cond 条件完全无效。请检查字典目录是否存在、")
            lines.insert(3, "     script_id 映射是否正确。")
            lines.insert(4, "=" * 58)
        elif rate < 0.3:
            lines.append(f"  [warn] 覆盖率偏低 ({rate*100:.1f}%)，多数样本 g 为零，"
                         f"条件作用有限")
        print("\n".join(lines), flush=True)
        return rate

    # -------------------------------------------------------------- getitem
    def __getitem__(self, idx):
        row = self.samples[idx]

        # img_id: 全局唯一样本键 (REPA DINO 特征缓存查表用)
        _m = re.search(r"(\d+)\.png", row['image_path'])
        if not _m:
            raise ValueError(f"Cannot parse img_id from {row['image_path']}")
        img_id = int(_m.group(1))

        if self.preload:
            latent = torch.from_numpy(self._latents[idx])
            img_t = torch.empty(0)
            if self.load_image and self._imgs is not None:
                a = self._imgs[idx].astype(np.float32) / 255.0 * 2.0 - 1.0
                img_t = torch.from_numpy(a).permute(2, 0, 1)
            skel_lat = torch.empty(0)
            if self._skel_latents is not None:
                skel_lat = torch.from_numpy(self._skel_latents[idx])
            aux_t = (torch.cat([torch.from_numpy(a[idx]) for a in self._aux_latents], 0)
                     if self._aux_latents else torch.empty(0))
        else:
            latent = self._get_latent(img_id)

            # 原始图 256 -> [-1,1] (优先 csv image_path, 支持变体分散在多个目录)
            img_t = torch.empty(0)
            if self.load_image:
                p = row["image_path"]
                full = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
                if not os.path.isfile(full) and self.img_root:
                    full = os.path.join(self.img_root, f"{img_id}.png")
                with Image.open(full) as im:
                    img = im.convert('RGB')
                img_t = (torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2.0 - 1.0)

            # skel latent (latent 条件) -> (C,32,32) float32
            skel_lat = torch.empty(0)
            if self._skel_id_to_shard:
                try:
                    sp, j = self._skel_id_to_shard[img_id]
                except KeyError as exc:
                    raise KeyError(f"skel latent not found for img_id={img_id}") from exc
                with np.load(sp) as shard:
                    skel_lat = torch.from_numpy(
                        np.array(shard["latents"][j], copy=True)).float()

            # aux latents (moyi 式辅助目标通道) -> (K*C,32,32)
            _aux_parts = []
            for _mp in self._aux_id_to_shard:
                _sp, _j = _mp[img_id]
                with np.load(_sp) as _shard:
                    _aux_parts.append(torch.from_numpy(
                        np.array(_shard["latents"][_j], copy=True)).float())
            aux_t = torch.cat(_aux_parts, 0) if _aux_parts else torch.empty(0)

        # 标准字形 latent g(甲2): 按 (script_id, char) 查标准字形 latent; 缺失给零(保 collate 一致)
        if self._glookup is not None:
            script_id = int(row['script_id'])
            char = row.get('character', '')
            # 训练时在该书体的可用字体间随机选 = 免费增广；推理时固定默认字体保证可复现
            gv = (self._glookup.get(script_id, char, random=self.is_train)
                  if char else None)
            if gv is not None:
                g_t = gv.float().contiguous()   # (4,32,32)
                self._g_hit += 1
            else:
                g_t = torch.zeros(self.latent_channels, self.latent_spatial, self.latent_spatial)
                self._g_miss += 1
        else:
            g_t = torch.zeros(0)

        return {
            'latent': latent,
            'img_id': img_id,
            'image': img_t,
            'canny': torch.empty(0),
            'skeleton': torch.empty(0),
            'skel_latent': skel_lat,
            'aux_latents': aux_t,
            'y_callig': torch.tensor(
                _map_callig(int(row['calligrapher_id']), self._callig_map), dtype=torch.long),
            'y_script': torch.tensor(int(row['script_id']), dtype=torch.long),
            'y_char': torch.tensor(
                int(row.get('glyph_id', row['character_id'])), dtype=torch.long),
            'g': g_t,
        }
