import numpy as np
from glob import glob
from os import listdir
from os.path import join
from .base_dataset import BaseDataset

class CelebDF(BaseDataset):
    """
    Celeb-DF v2 Dataset proposed in "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics".
    """

    def __init__(self, root, split, dataset_name='Celeb-DF', protocol='DI-IDD', balance=False):

        super().__init__(root, split, dataset_name, protocol)

        self.dataset_name = dataset_name
        self.categories = ['original', 'fake']

        self.root = join(root, 'Celeb-DF')

        images_ids = self.__get_images_ids()
        test_ids = self.__get_test_ids()
        train_ids = [images_ids[0] - test_ids[0],
                     images_ids[1] - test_ids[1],
                     images_ids[2] - test_ids[2]]
        self.images, self.targets = self.__get_images(
            test_ids if split == "test" else train_ids, balance)
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."

        print(f"{split} Data from 'Celeb-DF' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

        self.transforms = self.get_transforms(dataset_name, split, protocol)

    def __get_images_ids(self):
        youtube_real = listdir(join(self.root, 'YouTube-real'))
        celeb_real = listdir(join(self.root, 'Celeb-real'))
        celeb_fake = listdir(join(self.root, 'Celeb-synthesis'))
        return set(youtube_real), set(celeb_real), set(celeb_fake)

    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.root, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def __get_images(self, ids, balance=False):
        real = list()
        fake = list()
        # YouTube-real
        for _ in ids[0]:
            real.extend(glob(join(self.root, 'YouTube-real', _, '*.png')))  # [:2]
        # Celeb-real
        for _ in ids[1]:
            real.extend(glob(join(self.root, 'Celeb-real', _, '*.png')))  # [:2]
        # Celeb-synthesis
        for _ in ids[2]:
            fake.extend(glob(join(self.root, 'Celeb-synthesis', _, '*.png')))  # [:2]
        print("Real: {}, Fake: {}".format(len(real), len(fake)))
        if balance:
            fake = np.random.choice(fake, size=len(real), replace=False)
            print("After Balance | Real: {}, Fake: {}".format(len(real), len(fake)))
        real_tgt = [0] * len(real)
        fake_tgt = [1] * len(fake)
        return [*real, *fake], [*real_tgt, *fake_tgt]
