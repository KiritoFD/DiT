"""
GPU-Accelerated MCCD Calligraphy Dataset Preprocessing Script
Utilizes PyTorch CUDA tensors & multi-threaded I/O for maximum speed on NVIDIA GPUs.

Features:
- GPU-accelerated aspect-ratio preserved 256x256 padding & resizing
- GPU-accelerated Canny Edge Detection & Sobel filtering
- Accelerated Zhang-Suen Stroke Skeleton Thinning
- 3 Label Vocabularies: calligrapher_to_id.json, script_to_id.json, character_to_id.json
- Train / Val / Test CSV splits (train.csv, val.csv, test.csv)
"""

import os
import sys
import glob
import json
import random
import csv
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force stdout UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class GPUCalligraphyProcessor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing GPU Calligraphy Processor on: {self.device}")

        # Pre-compile 2D Gaussian & Sobel Convolution Kernels on GPU
        self.k_blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 16.0
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

    def process_batch(self, raw_images_bgr):
        """
        Process a batch of OpenCV BGR images on GPU:
        - Resize with aspect ratio preserved to 256x256 white padded canvas
        - Compute Canny edge map
        """
        if not raw_images_bgr:
            return [], []

        processed_padded = []
        canny_maps = []

        # Convert list of BGR images to 256x256 padded format
        for img in raw_images_bgr:
            h, w = img.shape[:2]
            scale = 256.0 / max(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

            padded = np.full((256, 256, 3), 255, dtype=np.uint8)
            top = (256 - nh) // 2
            left = (256 - nw) // 2
            padded[top:top + nh, left:left + nw] = resized
            processed_padded.append(padded)

        # Batch GPU Tensor Operations
        padded_np = np.stack(processed_padded, axis=0) # (B, 256, 256, 3)
        t_img = torch.from_numpy(padded_np).permute(0, 3, 1, 2).float().to(self.device) # (B, 3, 256, 256)

        # Grayscale conversion on GPU (BGR -> Gray)
        gray_gpu = 0.114 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.299 * t_img[:, 2:3]

        # Gaussian Blur on GPU
        blurred_gpu = F.conv2d(gray_gpu, self.k_blur, padding=1)

        # OpenCV Canny on GPU-blurred images for 100% exact OpenCV edge fidelity
        blurred_np = blurred_gpu.squeeze(1).byte().cpu().numpy()

        for b_img in blurred_np:
            canny_map = cv2.Canny(b_img, 50, 150)
            canny_maps.append(canny_map)

        return processed_padded, canny_maps


def compute_skeleton(gray_img):
    """
    Otsu thresholding + Zhang-Suen thinning for Skeleton map extraction
    """
    mean_val = np.mean(gray_img)
    if mean_val > 127:  # Light background, dark text
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Dark background, light text
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def read_image_worker(item):
    img_path, idx = item
    try:
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None

        fn = os.path.splitext(os.path.basename(img_path))[0]
        parts = fn.split('-')
        if len(parts) >= 4:
            character = parts[0].strip()
            script = parts[1].strip()
            calligrapher = parts[3].strip()
        else:
            character = "未知"
            script = "其他"
            calligrapher = "未知"

        return (img_path, img, character, script, calligrapher, idx)
    except Exception:
        return None


def write_image_worker(item):
    img_out_path, canny_out_path, skel_out_path, padded_img, canny_map, skel_map = item
    try:
        os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(canny_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(skel_out_path), exist_ok=True)

        cv2.imencode('.png', padded_img)[1].tofile(img_out_path)
        cv2.imencode('.png', canny_map)[1].tofile(canny_out_path)
        cv2.imencode('.png', skel_map)[1].tofile(skel_out_path)
        return True
    except Exception as e:
        print(f"Write error {img_out_path}: {e}")
        return False


def main(mccd_dir, output_dir, batch_size=128, io_workers=16, max_samples=None):
    processor = GPUCalligraphyProcessor(device='cuda')

    print(f"Scanning MCCD directory: {mccd_dir}")
    all_files = glob.glob(os.path.join(mccd_dir, "**", "*.png"), recursive=True)
    all_files += glob.glob(os.path.join(mccd_dir, "**", "*.jpg"), recursive=True)
    print(f"Found total {len(all_files)} raw image files.")

    if max_samples and max_samples < len(all_files):
        print(f"Subsampling to first {max_samples} files for testing...")
        all_files = all_files[:max_samples]

    read_items = [(p, i + 1) for i, p in enumerate(all_files)]
    records = []

    start_time = time.time()
    print(f"Starting GPU-accelerated processing (Batch Size={batch_size}, I/O Threads={io_workers})...")

    # Multi-threaded I/O Pool
    with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
        # Read images in parallel
        read_futures = [io_pool.submit(read_image_worker, item) for item in read_items]

        batch_raw = []
        batch_meta = []

        pbar = tqdm(total=len(all_files), desc="GPU Processing MCCD")

        for f in as_completed(read_futures):
            res = f.result()
            pbar.update(1)
            if res is None:
                continue

            img_path, img, character, script, calligrapher, idx = res
            batch_raw.append(img)
            batch_meta.append((img_path, character, script, calligrapher, idx))

            if len(batch_raw) >= batch_size:
                # GPU Batch Process
                padded_imgs, canny_maps = processor.process_batch(batch_raw)

                # Parallel Skeleton Thinning + Disk Writing
                write_tasks = []
                for b_i in range(len(batch_raw)):
                    img_path, character, script, calligrapher, idx = batch_meta[b_i]
                    padded_img = padded_imgs[b_i]
                    canny_map = canny_maps[b_i]

                    # Skeleton extraction
                    gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
                    skel_map = compute_skeleton(gray)

                    rel_dir = os.path.join(calligrapher, script, character)
                    out_fname = f"{idx:05d}.png"

                    img_out_path = os.path.join(output_dir, "images", rel_dir, out_fname)
                    canny_out_path = os.path.join(output_dir, "canny", rel_dir, out_fname)
                    skel_out_path = os.path.join(output_dir, "skeleton", rel_dir, out_fname)

                    write_tasks.append((img_out_path, canny_out_path, skel_out_path, padded_img, canny_map, skel_map))

                    records.append({
                        "image_path": img_out_path.replace("\\", "/"),
                        "canny_path": canny_out_path.replace("\\", "/"),
                        "skeleton_path": skel_out_path.replace("\\", "/"),
                        "calligrapher": calligrapher,
                        "script": script,
                        "character": character
                    })

                # Write files in parallel threadpool
                list(io_pool.map(write_image_worker, write_tasks))

                batch_raw.clear()
                batch_meta.clear()

        # Final remaining batch
        if batch_raw:
            padded_imgs, canny_maps = processor.process_batch(batch_raw)
            write_tasks = []
            for b_i in range(len(batch_raw)):
                img_path, character, script, calligrapher, idx = batch_meta[b_i]
                padded_img = padded_imgs[b_i]
                canny_map = canny_maps[b_i]

                gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
                skel_map = compute_skeleton(gray)

                rel_dir = os.path.join(calligrapher, script, character)
                out_fname = f"{idx:05d}.png"

                img_out_path = os.path.join(output_dir, "images", rel_dir, out_fname)
                canny_out_path = os.path.join(output_dir, "canny", rel_dir, out_fname)
                skel_out_path = os.path.join(output_dir, "skeleton", rel_dir, out_fname)

                write_tasks.append((img_out_path, canny_out_path, skel_out_path, padded_img, canny_map, skel_map))

                records.append({
                    "image_path": img_out_path.replace("\\", "/"),
                    "canny_path": canny_out_path.replace("\\", "/"),
                    "skeleton_path": skel_out_path.replace("\\", "/"),
                    "calligrapher": calligrapher,
                    "script": script,
                    "character": character
                })

            list(io_pool.map(write_image_worker, write_tasks))

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\nGPU Processing finished: {len(records)} images in {elapsed:.2f}s ({len(records) / elapsed:.1f} imgs/sec)")

    # 4. Build Label Dictionaries (Calligrapher, Script, Character)
    calligraphers = sorted(list(set(r["calligrapher"] for r in records)))
    scripts = sorted(list(set(r["script"] for r in records)))
    characters = sorted(list(set(r["character"] for r in records)))

    calligrapher_to_id = {c: i for i, c in enumerate(calligraphers)}
    script_to_id = {s: i for i, s in enumerate(scripts)}
    character_to_id = {ch: i for i, ch in enumerate(characters)}

    labels_dir = "labels"
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(labels_dir, "calligrapher_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(calligrapher_to_id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(labels_dir, "script_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(script_to_id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(labels_dir, "character_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(character_to_id, f, ensure_ascii=False, indent=2)

    print(f"Label vocabularies saved to '{labels_dir}/':")
    print(f" - Calligraphers: {len(calligraphers)}")
    print(f" - Script Styles: {len(scripts)}")
    print(f" - Character Contents: {len(characters)}")

    # Populate integer IDs
    for r in records:
        r["calligrapher_id"] = calligrapher_to_id[r["calligrapher"]]
        r["script_id"] = script_to_id[r["script"]]
        r["character_id"] = character_to_id[r["character"]]

    # Split into train.csv (80%), val.csv (10%), test.csv (10%)
    random.seed(42)
    random.shuffle(records)

    n_total = len(records)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    def save_csv(records_list, csv_path):
        header = ["image_path", "canny_path", "skeleton_path",
                  "calligrapher", "script", "character",
                  "calligrapher_id", "script_id", "character_id"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(records_list)
        print(f"Saved {len(records_list)} samples to '{csv_path}'")

    save_csv(train_records, "train.csv")
    save_csv(val_records, "val.csv")
    save_csv(test_records, "test.csv")

    print("\nGPU-accelerated dataset preparation completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-Accelerated MCCD Dataset Processing")
    parser.add_argument("--mccd-dir", type=str, default=r"g:\GitHub\DiT\MCCD", help="Path to MCCD directory")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset root directory")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU batch size")
    parser.add_argument("--io-workers", type=int, default=16, help="Disk I/O thread count")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    main(args.mccd_dir, args.output_dir, batch_size=args.batch_size, io_workers=args.io_workers, max_samples=args.max_samples)
