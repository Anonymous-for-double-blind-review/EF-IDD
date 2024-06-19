import numpy as np
from glob import glob
from os import listdir
from os.path import join
from .abstract_dataset import AbstractDataset

class FFIW(AbstractDataset):

    def __init__(self, root, split):
        self.dataset_name = 'FFIW'

        self.root = join(root, 'FFIW', split)
        self.images = []
        self.targets = []

        real = glob(join(self.root, '*', '*.png'))
        fake = glob(join(self.root, '*', '*.png'))

        self.images += real
        self.targets += [0] * len(real)

        self.images += fake
        self.targets += [1] * len(fake)

        print("Data from 'FFIW' loaded.\n")
        print("Dataset contains {} images.".format(len(self.images)))
        self.transform = self.get_transform(split)

