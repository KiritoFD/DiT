# 设置可见
# 一个epoch是1929396/768=2512 step, 5 epoch 存一下, 是 125611 step , ckpt-every 这么来的
#python train_moyun1.py \
# export TORCH_DISTRIBUTED_DEBUG=INFO
# set PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=5
echo $CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
export MKL_THREADING_LAYER=sequential

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 train_moyun2_diffusion_repa.py \
--model moyun-4channel-cmp1 \
--data-path /mnt/ssd/dataset/calligraphy/3312345/256png_SAM_small/training_set \
--results-dir /mnt/ssd/paper/cvpr2025/moyun2/model/model_cmp_moyun_small \
--feature-path /mnt/ssd/paper/cvpr2025/moyun2/dataset/256png_SAM_feature \
--log-every 5 \
--epochs 80000 \
--ckpt-every 9000 \
--global-batch-size 384 \
--learning-rate 1e-4 \
--if-rope 0 \
--if-rope-residual 0 \
--device 0 \
--num-classes 4793 \
--use_12channel 0 \
--custom_zero 0 \
--without-t 0 \
--charactor-x 0 \
--font-x 0 \
--calligrapher-x 0 