import cv2
from torch.utils.data import Dataset
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

albu_transform_list = [A.RandomBrightnessContrast( brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), brightness_by_max=True, p=1.0),
                       A.ImageCompression(quality_lower=1, quality_upper=99, p=1.0),
                       A.GaussianBlur(blur_limit=(3, 11), p=1.0),
                       A.CLAHE(clip_limit=(1, 8), tile_grid_size=(8, 8), p=1.0),
                       A.RandomGamma(gamma_limit=(10, 150), eps=None, p=1.0),
                       A.ToGray(p=1.0),
                       A.ChannelShuffle(p=1.0)
                       ]

class BaseDataset(Dataset):

    def __init__(self, root, split, dataset_name, protocol):

        self.root = root
        self.dataset_name = dataset_name

        self.images = []
        self.targets = []

        self.label_dict = {'0': 0, '1': 1}
        if protocol =='TI-IDD' :
            indices = os.path.join(self.root, 'TI-IDD', self.dataset_name, "{}.txt".format(split))
        else:
            indices = os.path.join(self.root, self.dataset_name, "{}.txt".format(split))

        with open(indices, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label_path = line.strip().split(',')
                self.images.append(label_path[1])
                self.targets.append(self.label_dict[label_path[0]])
        self.transforms = self.get_transforms(dataset_name, split, protocol)

    @staticmethod
    def get_transforms(dataset_name, split, protocol):

        normalize = A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        if split == 'test':
            transforms = A.Compose([
                A.Resize(384, 384),
                normalize,
                ToTensorV2()
            ])
        else:
            if  protocol == 'TI-IDD' or  dataset_name == 'ForgeryNet':
                transforms = A.Compose([
                    A.Resize(384, 384),
                    A.OneOf([
                        A.OneOf([
                                A.SomeOf(transforms=albu_transform_list, n=2, p=1.0),
                                A.SomeOf(transforms=albu_transform_list, n=3, p=1.0),
                                A.SomeOf(transforms=albu_transform_list, n=4, p=1.0)],
                                p=0.98),

                        A.OneOf(albu_transform_list,
                                p=0.01)],
                        p=0.99),
                    ToTensorV2(),
                    normalize
                ])
            else:
                transforms = A.Compose([
                    A.Resize(384, 384),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 5), p=-0.5)], p=0.1),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    ToTensorV2(),
                    normalize
                ])
        return transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # get data information.
        label  = self.targets[index]
        img_path = self.images[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_img = self.transforms(image=img)['image']

        return aug_img, label
