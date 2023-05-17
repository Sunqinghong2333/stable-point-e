yyy

srun -p DBX16 --quotatype=spot --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 \
-x SH-IDC1-10-198-8-[107,238,255,244,246,247,248,249,250,252,153,240],SH-IDC1-10-198-9-[1,12,10] \
python train_ae.py \
--train_batch_size 8 \
--val_batch_size 8 \
--val_freq 1000 \
--dataset_path /mnt/lustre/sunqinghong1/workspace/diffusion-point-cloud/datasets/shapenet.hdf5 
# > /dev/null &
