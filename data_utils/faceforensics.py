
import numpy as np
from .abstract_dataset import AbstractDataset
import glob
import os
import json


class FaceForensics(AbstractDataset):

    def __init__(self, root, split, domain_index,dataset_name='FF++', protocol='DI-IDD', balance=False):
        self.dataset_name = dataset_name
        self.domain_index = domain_index
        self.root = os.path.join(root, 'FaceForensics++')
        indices = os.path.join(self.root, split + ".json")
        real_path = os.path.join(self.root, 'original_sequences', 'youtube', 'videos')
        face_forgery = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        fake_paths = [os.path.join(self.root, 'manipulated_sequences', ff, 'videos') for ff in face_forgery]

        fake_num = len(fake_paths)
        with open(indices, 'r', encoding='utf-8') as f:
            folder_indices = json.load(f)

        self.images = []
        self.targets = []

        for i in folder_indices:
            real = glob.glob(real_path + '/' + i[0] + '/*.png')  # [:2]
            real += glob.glob(real_path + '/' + i[1] + '/*.png')  # [:2]

            self.images.extend(real)
            self.targets.extend([0] * len(real))

            for fake_path in fake_paths:
                fake_0 = glob.glob(fake_path + '/' + i[0] + '_' + i[1] + '/*.png')  # [:2]
                fake_1 = glob.glob(fake_path + '/' + i[1] + '_' + i[0] + '/*.png')  # [:2]

                if balance:
                    fake_0 = list(np.random.choice(fake_0, size=len(real) // (fake_num * 2), replace=False))

                    fake_1 = list(np.random.choice(fake_1, size=len(real) // (fake_num * 2), replace=False))
                fake = fake_0 + fake_1
                self.images.extend(fake)
                self.targets.extend([1] * len(fake))

        print("Data from 'FF++' loaded.\n")
        print("Dataset contains {} images.".format(len(self.images)))

        self.transforms = self.get_transforms(dataset_name, split, protocol)
