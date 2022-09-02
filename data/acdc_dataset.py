import numpy as np
from typing import Any, Dict, Tuple
import numpy as np
from torch.utils.data import Dataset

from data.acdc_utils import construct_samples_list, split_samples_list

# Computed on the labelled dataset
DATASET_MEAN = 67.27297657740893
DATASET_STD =  84.6606962344396 

NUM_PATIENTS = 100
TEST_PATIENTS = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
VAL_PATIENTS  = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
TRAIN_PATIENTS = [id for id in range(1, NUM_PATIENTS + 1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]

SPLIT_PATIENT_IDS = {'train': TRAIN_PATIENTS,
                     'val': VAL_PATIENTS,
                     'test': TEST_PATIENTS
                    }


class ACDCDatasetAlbu(Dataset):
    # The samples list and caches are class attributes and initialized during the first intantiation.
    all_samples = None
    img_cache = None
    gt_cache = None

    def __init__(self, root, transforms=None, split='train', subset_ratio=1.0, oversamle=False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        self.subset_ratio = subset_ratio

        if self.all_samples is None:
            all_samples, img_cache, gt_cache  = construct_samples_list(self.root)
            ACDCDatasetAlbu.all_samples = all_samples
            ACDCDatasetAlbu.img_cache = img_cache
            ACDCDatasetAlbu.gt_cache = gt_cache
        split_patient_ids = SPLIT_PATIENT_IDS[self.split]                                              
        self.samples = split_samples_list(self.all_samples, SPLIT_PATIENT_IDS[self.split])
        if 0.0 < subset_ratio < 1.0:
            subset_sample_indeces = np.random.choice(len(self.samples),
                                                     int(np.around(len(self.samples) * self.subset_ratio)),
                                                     replace=False).tolist()
            self.samples = [self.samples[id] for id in subset_sample_indeces]
            if oversamle:
                self.samples = self.samples * int(1/subset_ratio)
        self.num_patients = len(split_patient_ids)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        file_id, slice_id, img_path, target_path = self.samples[index]
            
        sample = self.img_cache[file_id][... , slice_id]
        target = self.gt_cache[file_id][... , slice_id]

        if self.transforms is not None:
            transformed = self.transforms(image=sample, mask=target)
            sample, target = transformed["image"], transformed["mask"]

        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        sample = np.expand_dims(sample, 0)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ACDCDatasetUnlabeleld(Dataset):
    # Deriving the unlabelled dataset from the labelled is very complicated. 
    # If they would be derived and use the same all_samples, what samples should this hold, only the labelled or unlabelled as well?
    # If both are instantiated in one namespace ACDCDataset.all_samples would hold the labelled list only.
    # Than ACDCDatasetUnlabeleld.all_samples shoudl hold the unlabelled list?
    all_samples = None
    img_cache = None

    def __init__(self, root, transform=None, split='train') -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.split = split

        if self.all_samples is None:
            all_samples, img_cache, _  = construct_samples_list(self.root, include_unlabelled=True)
            ACDCDatasetUnlabeleld.all_samples = all_samples
            ACDCDatasetUnlabeleld.img_cache = img_cache
        
        self.samples = split_samples_list(self.all_samples, SPLIT_PATIENT_IDS[self.split])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            np.ndarray: (sample)
        """
        file_id, slice_id, img_path, target_path = self.samples[index]            
        sample = self.img_cache[file_id][... , slice_id]

        if self.transform is not None:
            sample = self.transform(sample)
            
        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        if isinstance(sample, list):
            sample = [np.expand_dims(s['image'], 0) for s in sample]
        else:
            sample = np.expand_dims(sample['image'], 0)

        return sample, 0

    def __len__(self) -> int:
        return len(self.samples)


if __name__== "__main__":
    from tabulate import tabulate 

    train_set = ACDCDatasetAlbu("~/data/acdc/training", subset_ratio=0.1)
    print(tabulate(train_set.samples))
    print ("Train len =", len(train_set))
    val_set = ACDCDatasetAlbu("~/data/acdc/training", split='val')
    print ("Val len =", len(val_set))
    test_set = ACDCDatasetAlbu("~/data/acdc/training", split='test')
    print ("Test len =", len(test_set))

    unlabelled = ACDCDatasetUnlabeleld("~/data/acdc/training")
    print ("Unlabelled len =", len(unlabelled))
    unlabelled_val = ACDCDatasetUnlabeleld("~/data/acdc/training", split='val')
    print ("Unlabelled len =", len(unlabelled_val))