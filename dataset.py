import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

class MCCDDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size=256, load_maps=True,
                 load_canny=None, load_skel=None, is_train=False, use_glyph_cond=False):
        """
        csv_file: path to train.csv / val.csv
        root_dir: the root directory where 'images', 'canny', 'skeleton' are stored (e.g. 'dataset')
        load_maps: If False, skips opening canny and skeleton images for 3x CPU & Disk speedup!
        load_canny / load_skel: Independent switches for canny/skeleton maps.
                                 If None, both inherit from load_maps.
        is_train: If True, applies data augmentation (RandomAffine) for robust training.
        use_glyph_cond: If True, attach standard-glyph latent g by (script_id, character).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.load_maps = load_maps
        self.load_canny = load_maps if load_canny is None else load_canny
        self.use_glyph_cond = bool(use_glyph_cond)
        if self.use_glyph_cond:
            from glyph_latent import get_glyph_lookup
            self._glookup = get_glyph_lookup()
        else:
            self._glookup = None
        self.load_skel = load_maps if load_skel is None else load_skel
        self.is_train = is_train
        
        self.samples = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
                
        # Main transform for images (RGB)
        # Note: We DO NOT use RandomHorizontalFlip for Chinese characters!
        transforms_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        self.img_transform = transforms.Compose(transforms_list)
        
        if self.load_canny or self.load_skel:
            # Transform for binary maps (Canny, Skeleton) -> [0, 1]
            self.map_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        
        # Random affine parameters are drawn once per item and applied to BOTH the
        # image and the structural maps, so the supervision stays spatially aligned.
        self.affine = transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)) if is_train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        
        img_path = row['image_path']
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        with Image.open(img_path) as img:
            img = img.convert('RGB')
        img_t = self.img_transform(img)
        
        canny_t = torch.empty(0)
        skeleton_t = torch.empty(0)
        if self.load_canny:
            if self.root_dir:
                canny_path = os.path.join(self.root_dir, row['canny_path'])
            else:
                canny_path = row['canny_path']
            with Image.open(canny_path) as canny:
                canny = canny.convert('L')
            canny_t = (self.map_transform(canny) > 0.5).float()
        if self.load_skel:
            if self.root_dir:
                skeleton_path = os.path.join(self.root_dir, row['skeleton_path'])
            else:
                skeleton_path = row['skeleton_path']
            with Image.open(skeleton_path) as skeleton:
                skeleton = skeleton.convert('L')
            skeleton_t = (self.map_transform(skeleton) > 0.5).float()
        
        # Apply the SAME random affine transform to the image and structural maps
        # so that canny/skeleton supervision stays pixel-aligned with the image.
        if self.affine is not None:
            angle, translate, scale, shear = self.affine.get_params(
                self.affine.degrees, self.affine.translate, self.affine.scale, self.affine.shear,
                img_t.shape[-2:]
            )
            img_t = F.affine(img_t, angle, translate, scale, shear,
                             interpolation=transforms.InterpolationMode.BILINEAR)
            if canny_t.numel() > 0:
                canny_t = F.affine(canny_t, angle, translate, scale, shear,
                                   interpolation=transforms.InterpolationMode.NEAREST)
            if skeleton_t.numel() > 0:
                skeleton_t = F.affine(skeleton_t, angle, translate, scale, shear,
                                      interpolation=transforms.InterpolationMode.NEAREST)
        
        # 标准字形 latent g(甲2): 按 (script_id, character) 查标准字形 latent; 缺失给零(保 collate 一致)
        g_t = torch.zeros(4, 32, 32)
        if self._glookup is not None:
            script_id = int(row['script_id'])
            char = row.get('character', '')
            gv = self._glookup.get(script_id, char) if char else None
            if gv is not None:
                g_t = gv.float().contiguous()
        else:
            g_t = torch.zeros(0)

        return {
            'image': img_t,
            'canny': canny_t,
            'skeleton': skeleton_t,
            'y_callig': torch.tensor(int(row['calligrapher_id']), dtype=torch.long),
            'y_script': torch.tensor(int(row['script_id']), dtype=torch.long),
            'y_char': torch.tensor(
                int(row.get('glyph_id', row['character_id'])), dtype=torch.long),
            'g': g_t,
        }
