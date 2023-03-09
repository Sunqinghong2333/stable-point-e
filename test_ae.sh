#!/usr/bin/env bash

srun -p DBX16 --quotatype=reserved --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 \
-x SH-IDC1-10-198-8-[107,238,255,244,246,247,248,249,250,252,153,240],SH-IDC1-10-198-9-[1,12,10] \
python test_ae.py \
--batch_size 1 \
--ckpt /mnt/lustre/sunqinghong1/workspace/diffusion-point-cloud/logs_ae/AE_2023_03_06__22_52_55/ckpt_0.000000_25000.pt \
--dataset_path /mnt/lustre/sunqinghong1/workspace/diffusion-point-cloud/datasets/shapenet.hdf5

# > /dev/null &