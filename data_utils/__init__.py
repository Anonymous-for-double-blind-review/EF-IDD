from .base_dataset import BaseDataset
from .faceforensics import FaceForensics
from .celeb_df import CelebDF
from .dfdc import DFDC
from .ffiw import FFIW
from .dffd import DFFD
from .openforensics import OpenForensics

dataset_dict = {'FF++': FaceForensics,
                'Celeb-DF': CelebDF,
                'DFDC-P': DFDC,
                'FFIW': FFIW,
                'DFFD': DFFD,
                'OpenForensics': OpenForensics
                }


def get_dataset(data_name, split='train'):
    if data_name not in dataset_dict.keys():
        raise Exception('unknown dataset')

    dataset = dataset_dict[data_name](root='./dataset', split=split)
    
    return dataset
