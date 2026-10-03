#!/usr/bin/env bash
# 导出梯度检查点相关的精确代码 (远端版), 供写删除补丁
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "########## [1] dit.py: use_checkpoint 参数与属性 ##########"
grep -n 'use_checkpoint' src/model/dit.py

echo
echo "########## [2] dit.py: checkpoint 分支全文 (2455~2560) ##########"
sed -n '2455,2560p' src/model/dit.py | cat -n | sed 's/^/   /'

echo
echo "########## [3] train.py: use_checkpoint 全部出现 ##########"
grep -n 'use_checkpoint\|grad_ckpt' src/train/train.py

echo
echo "########## [4] eval/model_io.py ##########"
grep -n 'use_checkpoint' src/eval/model_io.py

echo
echo "########## [5] cli.py ##########"
grep -n -A4 'use-checkpoint' src/train/cli.py

echo
echo "########## [6] 其它非 legacy 引用 ##########"
grep -rn 'use_checkpoint' src/model/modules.py src/model/*.py 2>/dev/null | grep -v legacy
