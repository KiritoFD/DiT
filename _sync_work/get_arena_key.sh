#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
D=exp-std/signal_arena_full
L=exp-std/logs_purestd/arena_latest.log

echo "########## 1. 关键日志行 ##########"
grep -aE '\[tri\]|\[enc\]|\[lat\]|\[guard\]|\[pack\]|\[auroc\]|\[vec\]|自检|PSNR=|\[inv\]|\[summary\]|\[poster\]' "$L" 2>/dev/null

echo
echo "########## 2. summary.txt ##########"
cat "$D/summary.txt" 2>/dev/null

echo
echo "########## 3. margin.csv ##########"
cat "$D/margin.csv" 2>/dev/null

echo
echo "########## 4. inversion 聚合 (每 vae×signal, n=16 次反演) ##########"
awk -F, 'NR>1{k=$1"|"$2; n[k]++; ps[k]+=$4; ss[k]+=$5; io[k]+=$6; ny[k]+=$7; fo[k]+=$8}
END{for(k in n) printf "%-12s %-10s PSNR=%6.2f SSIM=%7.4f inkIoU=%6.3f 伪影比=%7.4f 收敛比=%7.3f n=%d\n",
  substr(k,1,index(k,"|")-1), substr(k,index(k,"|")+1), ps[k]/n[k], ss[k]/n[k], io[k]/n[k], ny[k]/n[k], fo[k]/n[k], n[k]}' \
  "$D/inversion.csv" | sort

echo
echo "########## 5. 判据2 里的 raw_l1 单条样例(看收敛比) ##########"
head -3 "$D/inversion.csv"
