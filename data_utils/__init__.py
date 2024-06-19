from .abstract_dataset import AbstractDataset
from .faceforensics import FaceForensics
from .celeb_df import CelebDF
from .dfdc import DFDC
from .ffiw import FFIW

dataset_dict = {'FF++': FaceForensics,
                'CDF2': CelebDF,
                'DFDC-P': DFDC,
                'FFIW': FFIW}


def get_dataset(data_name, split='train'):
    if data_name not in dataset_dict.keys():
        raise Exception('unknown dataset')

    dataset = dataset_dict[data_name](root='./dataset', split=split)
    
    return dataset
