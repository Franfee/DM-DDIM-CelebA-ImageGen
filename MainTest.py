# -*- coding: utf-8 -*-
# @Time    : 2023/12/25
# @Author  : FanAnfei
# @python  : Python 3.9.12


import os
import torch
import math
from torchvision import utils
from Trainer import num_to_groups
from net.Unet import Unet
from net.GaussianDiffusion import GaussianDiffusion


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)

    print("establish denoising")
    denoising = Unet(dim=64, dim_mults=(1, 2, 4, 8))

    print("establish diffusion")
    img_size = 256//2
    # sampling_timesteps: number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    diffusion = GaussianDiffusion(denoising, image_size=img_size, timesteps=1000,  # number of steps
                                  sampling_timesteps=50, loss_type='l1'  # L1 or L2
                                  )
    
    # 
    print("loading diffusion")
    ckpt = torch.load("results/model-200.pt", map_location='cpu')
    diffusion.load_state_dict(ckpt['model'])

    with torch.no_grad():
        batches = num_to_groups(4, 1)
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n), batches))

        all_images = torch.cat(all_images_list, dim=0)
        utils.save_image(all_images, "results/test.png", nrow=int(math.sqrt(4)))