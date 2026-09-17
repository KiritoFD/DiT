from PIL import Image, ImageDraw

V11 = "v11_pretrain_M432_adaln4_fame_kxl_tj_px60/eval_samples_ctrl/step0020000/strict"
V12 = "v12_pretrain_S_cat_fame_kxl_tj_px60/eval_samples_ctrl/step0020000/strict"
IDX = [0, 1, 2, 3]

ims11 = [Image.open(f"{V11}/g{i}.png").convert("RGB") for i in IDX]
ims12 = [Image.open(f"{V12}/g{i}.png").convert("RGB") for i in IDX]
w, h = ims11[0].size
print("cell", w, h)

PAD, LBL = 8, 26
cols = len(IDX)
W = cols * w + (cols + 1) * PAD
H = 2 * h + 3 * PAD + 2 * LBL
canvas = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(canvas)

y = PAD
d.text((PAD, y + 6), "v11  M/2 (h=432)  step 20000", fill=(0, 0, 0))
y += LBL
for c, im in enumerate(ims11):
    canvas.paste(im, (PAD + c * (w + PAD), y))
y += h + PAD
d.text((PAD, y + 6), "v12  S/2 (h=384) + concat(callig, glyph_vec)  step 20000", fill=(0, 0, 0))
y += LBL
for c, im in enumerate(ims12):
    canvas.paste(im, (PAD + c * (w + PAD), y))

canvas.save("cmp_v11_v12_step20000.png")
print("saved", canvas.size)
