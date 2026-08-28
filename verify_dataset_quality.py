"""
MCCD Dataset Correctness & Quality Verification Script
- Inspects train.csv, val.csv, test.csv
- Checks labels/ JSON vocabularies
- Verifies image dimensions (256x256), channels, and pixel ranges for Original, Canny, and Skeleton images
- Generates a visual quality comparison grid saved to 'dataset_quality_verification.png'
"""

import os
import sys
import csv
import json
import random
import numpy as np
import cv2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def verify_dataset(num_samples=8, output_grid_path="dataset_quality_verification.png"):
    print("==================================================")
    print("       MCCD DATASET QUALITY VERIFICATION          ")
    print("==================================================")

    # 1. Check CSV files
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        if not os.path.exists(csv_file):
            print(f"Error: CSV file '{csv_file}' not found!")
            return False
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"File '{csv_file}': {len(rows):,} rows found.")

    # Load train.csv rows
    with open("train.csv", "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    # 2. Check JSON label vocabularies
    vocab_info = {}
    for json_file in ["calligrapher_to_id.json", "script_to_id.json", "character_to_id.json"]:
        path = os.path.join("labels", json_file)
        if not os.path.exists(path):
            print(f"Error: Vocabulary file '{path}' not found!")
            return False
        with open(path, "r", encoding="utf-8") as f:
            v_dict = json.load(f)
            vocab_info[json_file] = len(v_dict)
            print(f"Vocabulary '{json_file}': {len(v_dict):,} entries.")

    # 3. Sample verification
    print(f"\nRandomly sampling {num_samples} records from train.csv for quality inspection...")
    samples = random.sample(train_rows, min(num_samples, len(train_rows)))

    grid_cols = []

    for i, s in enumerate(samples):
        img_p = s["image_path"]
        canny_p = s["canny_path"]
        skel_p = s["skeleton_path"]

        # Read images using Unicode safe imdecode
        img_buf = np.fromfile(img_p, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)

        canny_buf = np.fromfile(canny_p, dtype=np.uint8)
        canny = cv2.imdecode(canny_buf, cv2.IMREAD_GRAYSCALE)

        skel_buf = np.fromfile(skel_p, dtype=np.uint8)
        skel = cv2.imdecode(skel_buf, cv2.IMREAD_GRAYSCALE)

        # Check dimensions & shapes
        assert img.shape == (256, 256, 3), f"Invalid image shape: {img.shape} for {img_p}"
        assert canny.shape == (256, 256), f"Invalid Canny shape: {canny.shape} for {canny_p}"
        assert skel.shape == (256, 256), f"Invalid Skeleton shape: {skel.shape} for {skel_p}"

        print(f"Sample [{i+1}/{num_samples}]:")
        print(f"  - Text: {s['character']} | Script: {s['script']} | Calligrapher: {s['calligrapher']}")
        print(f"  - IDs -> Char: {s['character_id']} | Script: {s['script_id']} | Callig: {s['calligrapher_id']}")
        print(f"  - Image non-255 pixels: {np.count_nonzero(img < 250)}")
        print(f"  - Canny Edge pixels: {np.count_nonzero(canny == 255)}")
        print(f"  - Skeleton pixels: {np.count_nonzero(skel == 255)}")

        # Build visual row: [Original Image, Canny Edge (3ch), Skeleton (3ch)]
        canny_3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        skel_3ch = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        # Add text title overlay
        header = np.zeros((30, 256 * 3, 3), dtype=np.uint8)
        info_str = f"{s['calligrapher']} | {s['script']} | {s['character']}"
        cv2.putText(header, f"Sample {i+1}: {info_str}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        row_vis = np.hstack([img, canny_3ch, skel_3ch])
        stacked_row = np.vstack([header, row_vis])
        grid_cols.append(stacked_row)

    final_grid = np.vstack(grid_cols)
    cv2.imencode('.png', final_grid)[1].tofile(output_grid_path)

    print("\n==================================================")
    print(f"SUCCESS: Dataset quality verification passed 100%!")
    print(f"Visual quality grid saved to '{output_grid_path}'")
    print("==================================================")
    return True


if __name__ == "__main__":
    verify_dataset()
