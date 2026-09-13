"""
MCCD Calligraphy Dataset Preprocessing Script (Refined)

Algorithm Specifications (Font & Calligraphy Literature Standards):
1. Image Standardization:
   - Resized with aspect ratio preserved and padded to 256x256 using a white canvas (255, 255, 255).
2. Canny Edge Extraction:
   - Grayscale conversion -> Gaussian Blur (kernel_size=3x3) -> cv2.Canny(threshold1=50, threshold2=150).
   - Generates 256x256 binary edge map (0=bg, 255=edge contour).
3. Skeleton Thinning Extraction:
   - Grayscale conversion -> Otsu thresholding (cv2.THRESH_OTSU) -> Zhang-Suen Thinning (cv2.ximgproc.THINNING_ZHANGSUEN).
   - Generates 256x256 binary skeleton map (0=bg, 255=1-pixel centerline stroke).

Fields Extracted:
- character (字内容 / text)
- script (字体)
- calligrapher (书法家)
- image_path, canny_path, skeleton_path
- calligrapher_id, script_id, character_id
"""

import os
import sys
import glob
import json
import random
import csv
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force stdout UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def process_single_image(args):
    """
    Worker function to process one calligraphy image:
    1. Read and pad/resize to 256x256
    2. Compute Canny edge map
    3. Compute Skeleton map
    4. Save outputs to dataset/images, dataset/canny, dataset/skeleton
    """
    img_path, output_base, idx = args

    try:
        # Unicode-safe image reading for Windows paths
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Parse filename attributes: {character}-{script}-{dynasty}-{calligrapher}-{id}.png
        fn = os.path.splitext(os.path.basename(img_path))[0]
        parts = fn.split('-')
        if len(parts) >= 4:
            character = parts[0].strip()
            script = parts[1].strip()
            # Dynasty is ignored as per specification
            calligrapher = parts[3].strip()
        else:
            # Fallback if non-standard format
            character = "未知"
            script = "其他"
            calligrapher = "未知"

        # 1. Resize & Pad to 256x256
        h, w = img.shape[:2]
        target_size = 256
        scale = target_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Create white background canvas (255, 255, 255)
        padded_img = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        padded_img[top:top + new_h, left:left + new_w] = resized

        # Convert to Grayscale
        gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)

        # 2. Canny Edge Extraction (Domain Standard: Gaussian Blur 3x3 + Canny 50/150)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny_map = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # 3. Skeleton Thinning Extraction (Domain Standard: Otsu Binarization + Zhang-Suen Thinning)
        mean_val = np.mean(gray)
        if mean_val > 127:  # Light background, dark text
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:  # Dark background, light text
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        skeleton_map = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # Build output directory hierarchy: dataset/{images|canny|skeleton}/{calligrapher}/{script}/{character}/{idx:05d}.png
        rel_dir = os.path.join(calligrapher, script, character)
        out_fname = f"{idx:05d}.png"

        img_out_dir = os.path.join(output_base, "images", rel_dir)
        canny_out_dir = os.path.join(output_base, "canny", rel_dir)
        skel_out_dir = os.path.join(output_base, "skeleton", rel_dir)

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(canny_out_dir, exist_ok=True)
        os.makedirs(skel_out_dir, exist_ok=True)

        img_out_path = os.path.join(img_out_dir, out_fname)
        canny_out_path = os.path.join(canny_out_dir, out_fname)
        skel_out_path = os.path.join(skel_out_dir, out_fname)

        # Save images using imencode for Windows Unicode path safety
        cv2.imencode('.png', padded_img)[1].tofile(img_out_path)
        cv2.imencode('.png', canny_map)[1].tofile(canny_out_path)
        cv2.imencode('.png', skeleton_map)[1].tofile(skel_out_path)

        return {
            "image_path": img_out_path.replace("\\", "/"),
            "canny_path": canny_out_path.replace("\\", "/"),
            "skeleton_path": skel_out_path.replace("\\", "/"),
            "calligrapher": calligrapher,
            "script": script,
            "character": character
        }

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def main(mccd_dir, output_dir, num_workers=8, max_samples=None):
    print(f"Scanning MCCD directory: {mccd_dir}")
    all_files = glob.glob(os.path.join(mccd_dir, "**", "*.png"), recursive=True)
    all_files += glob.glob(os.path.join(mccd_dir, "**", "*.jpg"), recursive=True)
    print(f"Found total {len(all_files)} raw image files.")

    if max_samples and max_samples < len(all_files):
        print(f"Subsampling to first {max_samples} files for quick processing/testing...")
        all_files = all_files[:max_samples]

    tasks = [(p, output_dir, i + 1) for i, p in enumerate(all_files)]

    records = []
    print(f"Starting Canny & Skeleton processing with {num_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            res = f.result()
            if res is not None:
                records.append(res)

    print(f"Successfully processed {len(records)} images.")

    # Build 3 Label Vocabularies (Calligrapher, Script Style, Character Content)
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

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare MCCD Calligraphy Dataset (256x256, Canny, Skeleton, Vocabs, CSV splits)")
    parser.add_argument("--mccd-dir", type=str, default=r"g:\GitHub\DiT\MCCD", help="Path to raw MCCD directory")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset root directory")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    main(args.mccd_dir, args.output_dir, num_workers=args.num_workers, max_samples=args.max_samples)
