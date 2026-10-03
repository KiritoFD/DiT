# -*- coding: utf-8 -*-
"""1) 验证赵孟/赵孟頫是否同一人（六体字面重叠/图片重叠）
2) 检查遗漏的耳熟能详大家是否藏在数据里
3) 输出全量 top 200 书家中疑似人名的项
"""
import csv
from collections import Counter, defaultdict

# ---- 载入数据 ----
rows = []
with open("train.csv", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

c = Counter(x["calligrapher"] for x in rows)

# ---- 1) 赵孟 vs 赵孟頫 ----
print("=" * 60)
print("1) 赵孟 / 赵孟頫 六体重叠验证")
zm6 = defaultdict(set)  # 六体字符 -> 图片集合
for x in rows:
    if x["calligrapher"] in ("赵孟", "赵孟頫") and x["script"] == "六体":
        zm6[x["character"]].add(x["image_path"])

a_chars = set(k for k, v in zm6.items() if any("赵孟/" in i for i in v))
# 更精确：按书家分开
zm_chars = set()
zmf_chars = set()
for x in rows:
    if x["calligrapher"] == "赵孟" and x["script"] == "六体":
        zm_chars.add(x["character"])
    if x["calligrapher"] == "赵孟頫" and x["script"] == "六体":
        zmf_chars.add(x["character"])

print(f"赵孟[六体] 独字={len(zm_chars)}  赵孟頫[六体] 独字={len(zmf_chars)}")
inter = zm_chars & zmf_chars
print(f"六体字符交集={len(inter)}  并集={len(zm_chars | zmf_chars)}")
if inter:
    print(f"交集示例: {sorted(inter)[:15]}")

# 图片级重叠：相同字符下是否有相同图片
img_by_char = defaultdict(set)
for x in rows:
    if x["calligrapher"] in ("赵孟", "赵孟頫") and x["script"] == "六体":
        img_by_char[x["character"]].add(x["image_path"])
dup_img = sum(1 for v in img_by_char.values() if len(v) < 5 and len(v) >= 1)
# 检查是否有完全相同的图片路径出现在两个 id 下
img2c = defaultdict(set)
for x in rows:
    if x["calligrapher"] in ("赵孟", "赵孟頫") and x["script"] == "六体":
        img2c[x["image_path"]].add(x["calligrapher"])
same_img = {k: v for k, v in img2c.items() if len(v) > 1}
print(f"两 id 共用相同图片数 = {len(same_img)}")
if same_img:
    print(f"共用图片示例: {list(same_img)[:5]}")

# 全 script 字面重叠
zm_all = set(x["character"] for x in rows if x["calligrapher"] == "赵孟")
zmf_all = set(x["character"] for x in rows if x["calligrapher"] == "赵孟頫")
print(f"全script: 赵孟独字={len(zm_all)} 赵孟頫独字={len(zmf_all)} 交集={len(zm_all & zmf_all)}")

# ---- 2) 遗漏大家检查 ----
print("=" * 60)
print("2) 遗漏的耳熟能详大家是否在数据里")
EXTRA = ["赵佶", "宋徽宗", "卫夫人", "王珣", "陆机", "王僧虔", "皇象", "索靖",
         "李叔同", "弘一", "唐太宗", "李世民", "武则天", "康熙", "乾隆",
         "刘邦", "曹操", "李白", "杜甫", "韩愈", "柳宗元", "朱熹", "岳飞",
         "文天祥", "董其昌", "八大山人", "成亲王", "翁方纲", "刘墉",
         "溥儒", "张大千", "齐白石", "徐悲鸿", "刘海粟"]
for name in EXTRA:
    hit = c.get(name, 0)
    if hit:
        print(f"  {name}: {hit}")

# ---- 3) top 200 中疑似人名 ----
print("=" * 60)
print("3) top 200 中可能是人名但不在著名名单里的")
famous = {"王羲之","苏轼","欧阳询","颜真卿","智永","米芾","褚遂良","黄庭坚","文徵明",
"孙过庭","怀素","李阳冰","祝允明","何绍基","赵之谦","邓石如","吴昌硕","李邕","王铎",
"金农","沈尹默","柳公权","鲜于枢","王献之","赵孟頫","赵孟","董其昌","唐寅","欧阳通",
"蔡襄","虞世南","李斯","薛稷","朱耷","张旭","伊秉绶","钟繇","于右任","黄易","康里巎巎",
"徐渭","刘墉","杨凝式","傅山","张芝","怀仁","翁同龢","石涛","沙孟海","郑板桥","康有为",
"启功","林散之"}
for name, n in c.most_common(200):
    if name in famous:
        continue
    # 过滤明显的非人名：含非人名的模式
    if len(name) >= 2 and name[0] not in "隶篆楷行草六甲骨金文":
        print(f"  {name}: {n}")
