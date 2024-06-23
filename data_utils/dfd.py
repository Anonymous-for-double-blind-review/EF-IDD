
from glob import glob
from .abstract_dataset import AbstractDataset
import os


class DFD(AbstractDataset):

    def __init__(self, root, split, domain_index, dataset_name='DFD', protocol='DI-IDD'):
        self.dataset_name = dataset_name
        self.domain_index = domain_index

        self.root = os.path.join(root, dataset_name, split)
        self.images = []
        self.targets = []

        real = glob(os.path.join(self.root, 'DeepFakeDetection', '*', '*.png'))
        fake = glob(os.path.join(self.root, 'actors', '*', '*.png'))

        self.images += real
        self.targets += [0] * len(real)

        self.images += fake
        self.targets += [1] * len(fake)

        print(f"{split} Data from 'DFD' loaded.\n")
        print("Dataset contains {} images.".format(len(self.images)))
        self.transforms = self.get_transforms(dataset_name, split, protocol)