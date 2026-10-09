import os
import shutil
import glob

base = "/root/Workspace/xy/DiT"
if os.path.exists(base):
    os.chdir(base)

print("=== 开始整理 4090 服务器上的仓库文件 ===")

# 创建目标归档目录
target_dirs = {
    "archive_otout": "archive/debug_otout",
    "archive_tarballs": "archive/tarballs",
    "archive_test_images": "archive/test_images",
    "archive_diagnostics": "archive/diagnostics",
    "archive_code_backups": "archive/code_backups",
    "archive_legacy_runners": "scripts/legacy_runners",
    "archive_legacy_scripts": "scripts/legacy_scripts",
    "logs_legacy_out": "logs/legacy_out"
}

for name, d in target_dirs.items():
    os.makedirs(os.path.join(base, d), exist_ok=True)

# 1. 清理空文件 0, 1, 2
for empty_f in ["0", "1", "2"]:
    fp = os.path.join(base, empty_f)
    if os.path.isfile(fp) and os.path.getsize(fp) == 0:
        os.remove(fp)
        print(f"  • 删除临时空文件: {empty_f}")

# 2. 归档 _otout* 临时目录
otout_dirs = glob.glob(os.path.join(base, "_otout*"))
for od in otout_dirs:
    if os.path.isdir(od):
        dst = os.path.join(base, target_dirs["archive_otout"], os.path.basename(od))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(od, dst)
        print(f"  • 归档调试目录: {os.path.basename(od)} -> {target_dirs['archive_otout']}")

# 3. 归档根目录测试图片
test_imgs = glob.glob(os.path.join(base, "test_*.png")) + glob.glob(os.path.join(base, "_otout_*.png"))
for ti in test_imgs:
    if os.path.isfile(ti):
        dst = os.path.join(base, target_dirs["archive_test_images"], os.path.basename(ti))
        shutil.move(ti, dst)
        print(f"  • 归档测试图片: {os.path.basename(ti)} -> {target_dirs['archive_test_images']}")

# 4. 归档根目录零散 tar/tgz/xz 压缩包
root_tars = [
    "example_100.tgz", "supplements.tar.gz", "latent_missing.tar.gz",
    "top10_latents_train_eval.tar.xz", "exp_milestones_eval200.tar.gz",
    "historical_strict_common.tar.gz"
]
for tf in root_tars:
    fp = os.path.join(base, tf)
    if os.path.isfile(fp):
        dst = os.path.join(base, target_dirs["archive_tarballs"], tf)
        shutil.move(fp, dst)
        print(f"  • 归档压缩包: {tf} -> {target_dirs['archive_tarballs']}")

# 5. 归档根目录诊断报表 txt/csv
diag_files = [
    "base_inventory.txt", "base_noaug_report.txt", "base_polarity_bad.csv",
    "base_polarity_report.txt", "eval_sparsity.txt", "fame_tj_kxl_stats.txt",
    "mccd_target_check.txt", "tongji_new_calligs.txt", "tongji_repr.txt",
    "unicalli_stats.txt", "latent_struct_probe_cache.npz"
]
for df in diag_files:
    fp = os.path.join(base, df)
    if os.path.isfile(fp):
        dst = os.path.join(base, target_dirs["archive_diagnostics"], df)
        shutil.move(fp, dst)
        print(f"  • 归档诊断文件: {df} -> {target_dirs['archive_diagnostics']}")

# 6. 归档输出日志 nohup / out / eval_log
log_files = glob.glob(os.path.join(base, "*.out")) + glob.glob(os.path.join(base, "*.nohup")) + glob.glob(os.path.join(base, "*.eval_log"))
for lf in log_files:
    if os.path.isfile(lf):
        dst = os.path.join(base, target_dirs["logs_legacy_out"], os.path.basename(lf))
        shutil.move(lf, dst)
        print(f"  • 归档运行日志: {os.path.basename(lf)} -> {target_dirs['logs_legacy_out']}")

# 7. 归档历史代码备份 .bak
bak_files = glob.glob(os.path.join(base, "*.bak*")) + glob.glob(os.path.join(base, "*.save"))
for bf in bak_files:
    if os.path.isfile(bf):
        dst = os.path.join(base, target_dirs["archive_code_backups"], os.path.basename(bf))
        shutil.move(bf, dst)
        print(f"  • 归档代码备份: {os.path.basename(bf)} -> {target_dirs['archive_code_backups']}")

# 8. 归档根目录单次运行脚本 run_*.sh 和 launch_*.sh
runner_shs = glob.glob(os.path.join(base, "run_*.sh")) + glob.glob(os.path.join(base, "launch_*.sh")) + [
    os.path.join(base, "contr_status.sh"), os.path.join(base, "two_stage_inner.sh"),
    os.path.join(base, "probe_loss_ablation.sh")
]
for rf in runner_shs:
    if os.path.isfile(rf):
        dst = os.path.join(base, target_dirs["archive_legacy_runners"], os.path.basename(rf))
        shutil.move(rf, dst)
        print(f"  • 归档运行脚本: {os.path.basename(rf)} -> {target_dirs['archive_legacy_runners']}")

# 9. 归档零散的历史单次测试 python 脚本
single_py = [
    "test_contr.py", "test_contr2.py", "patch_joint.py",
    "repro_user_union_eval.py", "train_deform_standalone.py", "train_joint_v35_1step.py"
]
for pyf in single_py:
    fp = os.path.join(base, pyf)
    if os.path.isfile(fp):
        dst = os.path.join(base, target_dirs["archive_legacy_scripts"], pyf)
        shutil.move(fp, dst)
        print(f"  • 归档单次测试脚本: {pyf} -> {target_dirs['archive_legacy_scripts']}")

print("\n✓ 4090 服务器根目录文件整理完成！")
