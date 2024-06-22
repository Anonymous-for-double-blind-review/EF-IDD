
from glob import glob
from os.path import join
from .base_dataset import BaseDataset

class FFIW(BaseDataset):

    def __init__(self, root, split, dataset_name='FFIW', protocol='DI-IDD'):
        super().__init__(root, split, dataset_name, protocol)
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

        print("{} Data from 'FFIW' loaded.\n")
        print("Dataset contains {} images.".format(len(self.images)))
        self.transforms = self.get_transforms(dataset_name, split, protocol)

