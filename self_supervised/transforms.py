from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image
import torch
import albumentations as A

class AlbuTransforms:
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        self.transform = A.Compose([
                A.RandomResizedCrop(crop_size, crop_size, scale=(min_scale, max_scale)),
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=color_jitter_prob),
                A.GaussianBlur(p=gaussian_prob),
                A.Solarize(p=solarization_prob),
                A.HorizontalFlip(p=horizontal_flip_prob),
                A.Normalize(mean=mean, std=std),
            ])

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(image=x)

    def __repr__(self) -> str:
        return str(self.transform)
