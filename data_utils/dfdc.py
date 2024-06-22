from glob import glob
from os.path import join
from .base_dataset import BaseDataset
import json

class DFDC(BaseDataset):

    def __init__(self, split, root, dataset_name, protocol):
        super().__init__(root, split, dataset_name, protocol)
        self.dataset_name = 'DFDC'
        self.root = join(root, 'DFDC')
        indices = join(self.root, "dataset.json")
        with open(indices, 'r', encoding='utf-8') as f:
            folder_indices = json.load(f)
        label_dict = {'real': 0, 'fake': 1}
        self.images = []
        self.targets = []
        for k, v in folder_indices.items():
            if v['set'] == split:
                path_list = glob(join(self.root, k.replace('.mp4', ''), '*'))  # [:2]
                self.images.extend(path_list)
                self.targets.extend([label_dict[v['label']]] * len(path_list))

        print("Data from 'dfdc' loaded.\n")
        print("Dataset contains {} images.\n".format(len(self.images)))
        self.transforms = self.get_transforms(dataset_name, split, protocol)
