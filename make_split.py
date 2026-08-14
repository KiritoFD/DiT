# -*- coding: utf-8 -*-
"""
基于官方 train/test 划分做最终切分（不做额外隔离）：
  - eval: 1000 张，从官方 test 固定 seed 抽
  - test: 10000 张，从官方 test 剩余抽
  - train: 官方 train 全部 + 官方 test 剩余 = 318715
产出 final_split.json: {"img_id": "train"/"test"/"eval"} (329715 条)
同时写 final_manifest_split.json（在 manifest 基础上加 final_split 字段）。
"""
import json, sys, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_N = 1000
TEST_N = 10000
SEED = 42

m = json.load(open("final_manifest.json", encoding="utf-8"))
print("total:", len(m))

# 官方 test 的 img_id
test_ids = [r["img_id"] for r in m if r["split"] == "test"]
print("official test count:", len(test_ids))

random.seed(SEED)
shuffled = test_ids[:]
random.shuffle(shuffled)

eval_ids = set(shuffled[:EVAL_N])
test_ids_final = set(shuffled[EVAL_N:EVAL_N + TEST_N])

split_of = {}
n_eval = n_test = n_train = 0
for r in m:
    i = r["img_id"]
    if i in eval_ids:
        split_of[i] = "eval"
        n_eval += 1
    elif i in test_ids_final:
        split_of[i] = "test"
        n_test += 1
    else:
        split_of[i] = "train"
        n_train += 1

print(f"train={n_train} test={n_test} eval={n_eval} total={n_train+n_test+n_eval}")

# 断言无重复
assert n_train + n_test + n_eval == len(m)

with open("final_split.json", "w", encoding="utf-8") as f:
    json.dump(split_of, f, ensure_ascii=False)
print("written final_split.json")

# 追加 final_split 到 manifest
split_key = {str(k): v for k, v in split_of.items()}
out = []
for r in m:
    r2 = dict(r)
    r2["final_split"] = split_key[str(r["img_id"])]
    out.append(r2)
with open("final_manifest_split.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("written final_manifest_split.json")
