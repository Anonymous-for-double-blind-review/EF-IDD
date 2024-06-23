from glob import glob
from os.path import join
from .abstract_dataset import AbstractDataset


class OpenForensics(AbstractDataset):

    def __init__(self, split, root, domain_index, dataset_name='OpenForensics', protocol='D-IDD'):

        self.dataset_name = dataset_name
        self.transform = domain_index

        self.root = join(root, 'OpenForensics')

        self.images = []
        self.targets = []

        real_list = glob(join(self.root, split, 'real','*', '*'))

        self.images += real_list
        self.targets += [0] * len(real_list)

        fake_list = glob(join(self.root, split, 'fake','*', '*'))

        self.images += fake_list
        self.targets += [1] * len(fake_list)
        self.transforms = self.get_transforms(dataset_name, split, protocol)
        print(f"{split} Data from 'OpenForensics' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.")
