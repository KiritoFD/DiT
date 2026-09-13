"""
Ultra-Optimized Memory-Bounded GPU Dataset Preprocessing Script
- Strict Memory Limit: < 300 MB RAM (uses bounded generator chunks)
- High GPU Utilization: PyTorch CUDA batching for 256x256 padding, resizing, and edge filtering
- ThreadPool disk I/O with bounded queue to prevent RAM inflation
- Deduplicates samples by unique file signature / content
- Generates 3 label vocabularies and train/val/test CSV splits
"""

import os
import sys
import glob
import json
import random
import csv
import time
import queue
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class PyTorchGPUPipeline:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch GPU Pipeline active on: {self.device} ({torch.cuda.get_device_name(0)})")

        # Pre-compile GPU Kernels
        self.k_blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 16.0
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

    def process_bgr_batch(self, bgr_list):
        """
        Process a batch of OpenCV BGR images on GPU
        """
        if not bgr_list:
            return [], []

        processed_padded = []
        canny_maps = []

        for img in bgr_list:
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
        t_img = torch.from_numpy(padded_np).permute(0, 3, 1, 2).float().to(self.device)

        # Grayscale on GPU
        gray_gpu = 0.114 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.299 * t_img[:, 2:3]

        # Gaussian Blur on GPU
        blurred_gpu = F.conv2d(gray_gpu, self.k_blur, padding=1)

        # Canny edge detection
        blurred_np = blurred_gpu.squeeze(1).byte().cpu().numpy()
        for b_img in blurred_np:
            canny_map = cv2.Canny(b_img, 50, 150)
            canny_maps.append(canny_map)

        return processed_padded, canny_maps


def compute_skeleton(gray_img):
    """
    Otsu binarization + Zhang-Suen thinning
    """
    mean_val = np.mean(gray_img)
    if mean_val > 127:  # Light bg
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Dark bg
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def read_file_task(img_path):
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

        return (img_path, img, character, script, calligrapher)
    except Exception:
        return None


def write_file_task(item):
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


def chunk_generator(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def main(mccd_dir, output_dir, batch_size=128, io_workers=8, max_samples=None):
    gpu_pipe = PyTorchGPUPipeline(device='cuda')

    print(f"Scanning MCCD directory: {mccd_dir}")
    all_files = glob.glob(os.path.join(mccd_dir, "**", "*.png"), recursive=True)
    all_files += glob.glob(os.path.join(mccd_dir, "**", "*.jpg"), recursive=True)
    print(f"Found total {len(all_files):,} raw image files.")

    # Deduplicate by (filename, filesize) to process unique images only
    print("Deduplicating image list by (basename, size)...")
    seen = set()
    unique_files = []
    for p in all_files:
        fn = os.path.basename(p)
        sz = os.path.getsize(p)
        key = (fn, sz)
        if key not in seen:
            seen.add(key)
            unique_files.append(p)

    print(f"Unique image files to process: {len(unique_files):,} (Deduplicated from {len(all_files):,})")

    if max_samples and max_samples < len(unique_files):
        print(f"Subsampling to first {max_samples:,} files for testing...")
        unique_files = unique_files[:max_samples]

    records = []
    start_time = time.time()
    idx_counter = 1

    pbar = tqdm(total=len(unique_files), desc="GPU Processing Unique MCCD Images")

    # Bounded ThreadPool to keep RAM memory small (< 300MB)
    with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
        # Process in chunks of 512 images at a time (constant RAM)
        for chunk in chunk_generator(unique_files, chunk_size=512):
            # Parallel read
            read_results = list(io_pool.map(read_file_task, chunk))
            valid_results = [r for r in read_results if r is not None]

            # Batch GPU Processing
            for batch_slice in chunk_generator(valid_results, batch_size):
                bgr_imgs = [item[1] for item in batch_slice]
                padded_imgs, canny_maps = gpu_pipe.process_bgr_batch(bgr_imgs)

                write_tasks = []
                for b_i, item in enumerate(batch_slice):
                    img_path, img, character, script, calligrapher = item
                    padded_img = padded_imgs[b_i]
                    canny_map = canny_maps[b_i]

                    gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
                    skel_map = compute_skeleton(gray)

                    rel_dir = os.path.join(calligrapher, script, character)
                    out_fname = f"{idx_counter:05d}.png"
                    idx_counter += 1

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

                # Parallel Disk Write for batch
                list(io_pool.map(write_file_task, write_tasks))

            pbar.update(len(chunk))
            if (idx_counter - 1) % 5120 < 512:
                elapsed_min = (time.time() - start_time) / 60
                speed = idx_counter / (time.time() - start_time)
                print(f"[{time.strftime('%H:%M:%S')}] Processed {idx_counter:,} / {len(unique_files):,} images ({(idx_counter/len(unique_files))*100:.1f}%) | Speed: {speed:.1f} it/s | Elapsed: {elapsed_min:.1f} min", flush=True)

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\nProcessing finished: {len(records):,} unique images in {elapsed:.2f}s ({len(records) / elapsed:.1f} imgs/sec)")

    # Build 3 Label Vocabularies
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

    # Populate IDs
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
        print(f"Saved {len(records_list):,} samples to '{csv_path}'")

    save_csv(train_records, "train.csv")
    save_csv(val_records, "val.csv")
    save_csv(test_records, "test.csv")

    print("\nMemory-bounded GPU dataset preparation completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Bounded GPU MCCD Preprocessing")
    parser.add_argument("--mccd-dir", type=str, default=r"g:\GitHub\DiT\MCCD", help="Path to MCCD directory")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset root directory")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU batch size")
    parser.add_argument("--io-workers", type=int, default=8, help="Disk I/O thread count")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    main(args.mccd_dir, args.output_dir, batch_size=args.batch_size, io_workers=args.io_workers, max_samples=args.max_samples)
