# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 12:10
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import os
from Trainer import Trainer
from net.Unet import Unet
from net.GaussianDiffusion import GaussianDiffusion

LOAD_CHECK_POINT = False


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)

    print("establish denoising")
    denoising = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    print("establish diffusion")
    img_size = 256//2
    # sampling_timesteps: number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    diffusion = GaussianDiffusion(denoising, image_size=img_size, timesteps=1000,  # number of steps
                                  sampling_timesteps=200, loss_type='l1'  # L1 or L2
                                  ).cuda()
    """
        size64 batch1 = 2.0G
        size64 batch2 = 2.1G
        size64 batch4 = 2.4G
        size64 batch8 = 2.9G
        size64 batch16 = 3.4G

        size128 batch1 = 2.0G
        size128 batch2 = 3.0G
        size128 batch8 = 5.2G
        size128 batch32 = 17.7G
    """

    # trainer [defaults: save_and_sample_every=100, num_samples=25]
    datasetsPath = "datasets/celeba_hq_256"
    trainer = Trainer(diffusion, datasetsPath, train_batch_size=32, train_lr=8e-5, num_samples=16, train_num_steps=20000)

    # train model
    if LOAD_CHECK_POINT:
        milestone = 200
        trainer.load(milestone)
    trainer.train()
