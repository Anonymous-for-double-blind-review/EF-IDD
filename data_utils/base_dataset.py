
import os
from .abstract_dataset import AbstractDataset


class BaseDataset(AbstractDataset):

    def __init__(self, root, split, domain_index, dataset_name, protocol):

        self.root = root
        self.dataset_name = dataset_name
        self.domain_index = domain_index

        self.images = []
        self.targets = []

        self.label_dict = {'0': 0, '1': 1}
        if protocol == 'TI-IDD':
            indices = os.path.join(self.root, 'TI-IDD', self.dataset_name, "{}.txt".format(split))
        else:
            indices = os.path.join(self.root, self.dataset_name, "{}.txt".format(split))

        with open(indices, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label_path = line.strip().split(',')
                self.images.append(label_path[1])
                self.targets.append(self.label_dict[label_path[0]])
        self.transforms = self.get_transforms(dataset_name, split, protocol)
