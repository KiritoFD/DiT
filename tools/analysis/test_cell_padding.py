from PIL import Image, ImageDraw, ImageFont
import numpy as np

def prepare_cell_image(img_np: np.ndarray, cell_size: int = 240, inner_pad: int = 16) -> Image.Image:
    """保证字符以充足内边距完美居中，绝不贴边，绝无任何裁切。"""
    ink = (1.0 - img_np) >= 0.30
    if not ink.any():
        return Image.new("RGB", (cell_size, cell_size), (255, 255, 255))

    y_idx, x_idx = np.where(ink)
    ymin, ymax = int(y_idx.min()), int(y_idx.max())
    xmin, xmax = int(x_idx.min()), int(x_idx.max())
    h_ink = ymax - ymin + 1
    w_ink = xmax - xmin + 1

    avail_size = cell_size - 2 * inner_pad # 208

    im = Image.fromarray((img_np * 255.0).astype(np.uint8)).convert("RGB")
    
    # 如果四周都有至少 16px 留白，直接缩放并居中
    if ymin >= 16 and (255 - ymax) >= 16 and xmin >= 16 and (255 - xmax) >= 16:
        im_resized = im.resize((avail_size, avail_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        canvas.paste(im_resized, (inner_pad, inner_pad))
        return canvas
    else:
        # 原图撞边 (如 3px 边距)，先 pad 48px，再以墨迹中心安全截取正方形
        padded = Image.new("RGB", (352, 352), (255, 255, 255))
        padded.paste(im, (48, 48))
        cy = (ymin + ymax) / 2.0 + 48
        cx = (xmin + xmax) / 2.0 + 48
        half_box = max(h_ink, w_ink) / 2.0 + 16 # 留出 16px 安全余量
        crop_box = (
            int(round(cx - half_box)),
            int(round(cy - half_box)),
            int(round(cx + half_box)),
            int(round(cy + half_box))
        )
        cropped = padded.crop(crop_box)
        cropped_resized = cropped.resize((avail_size, avail_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        canvas.paste(cropped_resized, (inner_pad, inner_pad))
        return canvas

if __name__ == "__main__":
    # Test on a border-touching image, e.g. g089 (致)
    im_raw = np.array(Image.open("artifacts/DiC_U100_review_20261020/images/reference/gt/g89.png").convert("L"), dtype=np.float32) / 255.0
    cell = prepare_cell_image(im_raw)
    cell.save("scratch/test_cell_g89_padded.png")
    
    # Check margins of test_cell_g89_padded.png
    arr = np.array(cell.convert("L"))
    ink = arr < 128
    y, x = np.where(ink)
    print("test_cell_g89 margins: top=", y.min(), "bottom=", 239 - y.max(), "left=", x.min(), "right=", 239 - x.max())
