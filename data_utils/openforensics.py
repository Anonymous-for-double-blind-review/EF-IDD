from glob import glob
from os.path import join
from .base_dataset import BaseDataset

class OpenForensics(BaseDataset):

    def __init__(self, split, root, transform, dataset_name='OpenForensics', protocol='D-IDD'):

        super().__init__(root, split, dataset_name, protocol)
        self.dataset_name = dataset_name

        self.root = join(root, 'OpenForensics')
        self.transform = transform    
        self.images = []
        self.targets = []

        real_list= glob(join(self.root, split, 'real','*', '*'))

        self.images  += real_list
        self.targets += [0] * len(real_list)

        fake_list= glob(join(self.root, split, 'fake','*', '*'))

        self.images  += fake_list
        self.targets += [1] * len(fake_list)
        self.transforms = self.get_transforms(dataset_name, split, protocol)
        print(f"{split} Data from 'OpenForensics' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.")
