
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
import albumentations as A
from mmseg.datasets import build_dataset
from mmcv.utils import Config


config_paths = {
    "custom": "data/config/custom.py",
    "acdc_mmseg": "data/config/acdc_mmseg.py",
    # For configs of other datasets that are supported by mmseg see: 
    # https://github.com/open-mmlab/mmsegmentation/tree/ae78cb9d53f1a17de7c9e89cfdc08aa95314fb4d/configs/_base_/datasets
}

class MMseg(Dataset):
    """ Wrapper class for mmsegmentation datasets.
    To use it with a custom semantic segmentation dataset see the documentation of mmseg's custom dataset
    https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.CustomDataset.

    Args:
            split (str, optional):  The dataset split, supports `train`, `val`, `test` or `interobserver`.
            dataset (str, optional): Dataset type supported by mmsegmentation. Defaults to "custom".
            transforms (albumentations.Compose, optional): Albumentations augmentation pipeline. Defaults to None. WARNING: It is recommended to use mmsegmentation's transform pipeline that is configured in data/config/<dataset>.py! Using albumentations augmentations is not efficient.
    """
    def __init__(self, 
                 split,
                 dataset="custom",
                 transforms=None
                 ):
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.config = MMseg.get_config(dataset)
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))
        self.num_classes = self.config.num_classes

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if isinstance(self.transforms, A.Compose):
            im = data["img"]
            seg = data["gt_semantic_seg"]
            transformed = self.transforms(image=im, mask=seg)
            im, seg = transformed["image"], transformed["mask"]
            # Convert images to channels_first mode, from albumentations' 2d grayscale images
            im = np.expand_dims(im, 0)
        else:
            # Implement augmentations using mmsegmentations transforms pipeline
            # The .data is needed if the last step of the pipeline is DefaultFormatBundle
            im = data["img"].data
            seg = data["gt_semantic_seg"].data

        return im, seg

    @staticmethod
    def get_config(dataset):
        config_path = config_paths[dataset]
        config = Config.fromfile(config_path)
        return config


    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm
    import time

    ds = MMseg("val", "custom")
    for i in range(100):
        im, seg = ds[i]
        print(im.shape, im.dtype, seg.shape, seg.dtype)

    loader = DataLoader(
        ds,
        batch_size=8,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    im_shapes = []
    seg_shapes = []
    start_time = time.time()
    for batch in tqdm(iter(loader)):
        im, seg = batch
        im_shapes.append(torch.tensor(im.shape))
        seg_shapes.append(torch.tensor(seg.shape))
    load_time = time.time() - start_time
    print(f"Loading the complete dataset split set took {load_time}s {load_time/len(ds)*1000}ms per sample", )
    
    print(torch.unique(torch.stack(im_shapes), dim=0))
    print(torch.unique(torch.stack(seg_shapes), dim=0))


