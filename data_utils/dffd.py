import numpy as np
from glob import glob
from os import listdir
from os.path import join
from .base_dataset import AbstractDataset


class DFFD(AbstractDataset):
    def __init__(self, root, split, domain_index, dataset_name='DFFD', protocol='DI-IDD'):

        self.dataset_name = 'DFFD'
        self.domain_index = domain_index
        self.root = join(root, 'DFFD')

        self.images = []
        self.targets = []

        celeba = glob(join(self.root, 'celeba', split, '*.png'))

        self.images += celeba
        self.targets += [0] * len(celeba)

        ffhq = glob(join(self.root, 'ffhq', split, '*.png'))

        self.images += ffhq
        self.targets += [0] * len(ffhq)

        fake_dirs = ['faceapp', 'pggan_v1', 'pggan_v2', 'stylegan_celeba', 'stylegan_ffhq', 'stargan']

        for i in fake_dirs:
            fake = glob(join(self.root, i , split, '*.png'))

            print(i, len(fake))

            self.images += fake
            self.targets += [1] * len(fake)
        self.transforms = self.get_transforms(dataset_name, split, protocol)
        print(f"{split} Data from 'DFFD' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.")



