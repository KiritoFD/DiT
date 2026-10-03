from PIL import ImageFont
p = "/root/Workspace/xy/DiT/tools/fonts/SimHei.ttf"
f20 = ImageFont.truetype(p, 20)
f14 = ImageFont.truetype(p, 14)
print("font ok", f20.getsize("标准字形DINO+PCA"))
print("w@", f14.getsize("CtrlNet GT skel 1px b192"))
