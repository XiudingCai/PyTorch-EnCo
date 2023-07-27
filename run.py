import os

sh = "python train.py --dataroot Z:/CodeShop/GAN/CycleGAN-and-pix2pix-master/datasets/horse2zebra" \
     " --name CITY_EnCo --model enco --nce_layers 3,7,13,18,24,28 --batch_size 1" \
     " --n_epochs 100 --n_epochs_decay 100 --num_threads 0 --lambda_IDT 10 --lambda_NCE 2" \
     " --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4" \
     " --warmup_epochs 20 --flip_equivariance True"

os.system(sh)
