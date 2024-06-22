import numpy as np
from glob import glob
from os import listdir
from os.path import join
from .base_dataset import BaseDataset


class DFFD(BaseDataset):


    def __init__(self, root, split, transform, dataset_name='DFFD', protocol='DI-IDD'):

        super().__init__(root, split, dataset_name, protocol)
        self.dataset_name = 'DFFD'
        self.transform = transform
        self.root = join(root, 'DFFD')

        self.images =  []
        self.targets = []

        celeba = glob(join(self.root, 'celeba', split, '*.png'))

        self.images +=  celeba
        self.targets += [0] * len(celeba)

        ffhq = glob(join(self.root, 'ffhq', split, '*.png'))

        self.images +=  ffhq
        self.targets += [0] * len(ffhq)

        fake_dirs = ['faceapp', 'pggan_v1', 'pggan_v2', 'stylegan_celeba', 'stylegan_ffhq', 'stargan']

        for i in fake_dirs:
            fake = glob(join(self.root, i , split, '*.png'))

            print(i, len(fake))

            self.images +=  fake
            self.targets += [1] * len(fake)
        self.transforms = self.get_transforms(dataset_name, split, protocol)
        print(f"{split} Data from 'DFFD' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.")



