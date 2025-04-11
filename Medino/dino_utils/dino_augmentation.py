import random
import numpy as np
import torch
from torchvision import transforms


def seed_everything(seed=42):
    """Function to set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class DataAugmentationDINO(object):
    """Class for DINO data augmentation."""

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=64):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
        ])

        normalize = transforms.Compose([
            transforms.Normalize([0.326, 0.326, 0.326], [0.228, 0.228, 0.228])
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                antialias=True,
                scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                antialias=True,
                scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                96,
                antialias=True,
                scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        """Apply the transformations to the input image."""
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return image, crops


class KNN_augmentation(object):
    """Class for DINO data augmentation."""

    def __init__(self, global_crops_scale=None, image_size=64, p=1.0, mean_std=None):
        self.p = p  # Probability of applying transformation

        if mean_std is None:
            normalize = transforms.Compose([
                transforms.Normalize(
                    [0.3261, 0.3261, 0.3261],
                    [0.2283, 0.2283, 0.2283])
            ])
        else:
            normalize = transforms.Compose([
                transforms.Normalize(
                    mean_std[0],
                    mean_std[1])
            ])

        if global_crops_scale is None:
            print("No global crop scale")
            self.global_transfo1 = transforms.Compose([
                normalize,
            ])
        else:
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    antialias=True,
                    scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                normalize,
            ])

    def __call__(self, image):
        """Apply the transformations to the input image."""
        if random.random() < self.p:
            return self.global_transfo1(image)
        else:
            return image
