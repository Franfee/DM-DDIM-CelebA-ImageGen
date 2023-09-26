# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 20:49
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import math
import torch

from torch.nn.utils import clip_grad_norm_

from torchvision import utils
from torch.utils.data import DataLoader
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from pathlib import Path
from multiprocessing import cpu_count
from einops import rearrange, reduce, repeat

from util.MyDataset import MyDataset, cycleDataLoader


def exists(x):
    return x is not None


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Trainer(object):
    def __init__(self, model, folder, *, train_batch_size=16, augment_horizontal_flip=True,
                 train_lr=1e-4, train_num_steps=100000, adam_betas=(0.9, 0.99),
                 save_and_sample_every=100, num_samples=25, results_folder='./results',
                 convert_image_to=None, calculate_fid=False, inception_block_idx=2048):
        super().__init__()

        # results_folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # model
        self.model = model

        # InceptionV3 for fid-score computation
        self.inception_v3 = None
        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = model.image_size

        # dataset and dataloader
        self.ds = MyDataset(folder, self.image_size,
                            augment_horizontal_flip=augment_horizontal_flip,
                            convert_image_to=convert_image_to)
        # cpu_count()
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=0)
        self.dl = cycleDataLoader(dl)

        # optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr, betas=adam_betas)

        # step counter state
        self.step = 0
        print("Trainer init complete.")

    @property
    def device(self):
        return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def save(self, milestone):
        print("called save.")
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        print("torch saved.")

    def load(self, milestone):
        print("called load.")
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)
        self.step = data['step']
        self.optimizer.load_state_dict(data['opt'])
        self.model.load_state_dict(data['model'])
        print("torch loaded.")

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim=0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.model.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
                                             (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):

        print("In train.")

        while self.step < self.train_num_steps:
            self.step += 1

            data = next(self.dl).to(self.device)

            # total_loss = 0.
            loss = self.model(data)
            # total_loss += loss.item()
            loss.backward()

            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"[in step: {self.step}/{self.train_num_steps}] total_loss = {loss.item()}")

            # checkpoint and sampling
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # sampling
                print(f"[in step: {self.step}/{self.train_num_steps}] sampling.")
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    all_images_list = list(map(lambda n: self.model.sample(batch_size=n), batches))

                all_images = torch.cat(all_images_list, dim=0)
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                 nrow=int(math.sqrt(self.num_samples)))
                # save model
                if self.step != 0 and self.step % (10 * self.save_and_sample_every) == 0:
                    print(f"[in step: {self.step}/{self.train_num_steps}] save model.")
                    self.save(milestone)

                # whether to calculate fid
                if exists(self.inception_v3):
                    fid_score = self.fid_score(real_samples=data, fake_samples=all_images)
                    print(f'fid_score: {fid_score}')

        print('training complete')
