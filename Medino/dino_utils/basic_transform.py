from random import randint, random
from typing import List, Dict, Callable
from typing import Tuple

import numpy as np
import torch
import torchvision
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torch import from_numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseTranformation(object):
    def __init__(self, keys: List[str]):
        if len(keys) < 1:
            raise ValueError('The number of data keys must be at least one.')

        self.keys = keys

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        pass


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms: List[BaseTranformation]):
        self.transforms = transforms

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class PaddVideo(BaseTranformation):
    def __init__(self, keys: List[str], num_frames: int):
        super().__init__(keys)
        self.frames_num = num_frames

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        for k in self.keys:
            input_dict[k] = uniform_temporal_subsample(x=input_dict[k],
                                                       num_samples=self.frames_num,
                                                       temporal_dim=0)
        return input_dict


class ToTensor(BaseTranformation):
    def __init__(self, keys: List[str], contiguous=False):
        self.contiguous = contiguous
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        if self.contiguous:
            for k in self.keys:
                input_dict[k] = np.ascontiguousarray(input_dict[k])

        for k in self.keys:
            input_dict[k] = from_numpy(input_dict[k])

        return input_dict


class Resize(BaseTranformation):
    def __init__(self, keys: List[str], size: Tuple):
        super().__init__(keys)
        self.size = size

    def resize_video(self, x):
        # Create transform to resize each frame
        transform = torchvision.transforms.Resize(self.size, antialias=True)
        # Loop through each frame and apply the transform
        resized_frame = transform(x)
        return resized_frame.repeat(3, 1, 1)

    def __call__(self, input_dict: Dict[str, torch.tensor]):
        for k in self.keys:
            resized = self.resize_video(x=input_dict[k])
            input_dict[k] = resized
        return input_dict


class Grayscale(BaseTranformation):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def to_grayscale(self, x):
        # Convert RGB frame to grayscale by averaging the channels
        grayscale_frame = x.mean(dim=0, keepdim=True)  # Averaging across the color channels
        return grayscale_frame

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        for k in self.keys:
            grayscale = self.to_grayscale(x=input_dict[k])
            input_dict[k] = grayscale
        return input_dict


class DinoAug(BaseTranformation):
    def __init__(self, keys: List[str], transform):
        self.transform = transform
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        for k in self.keys:
            input_dict[k] = self.transform(input_dict[k])
        return input_dict


class RandomSelect(BaseTranformation):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        for k in self.keys:
            x = input_dict[k]
            frame, channels, H, W = x.shape
            selected_frame = randint(0, frame - 1)
            selected_data = x[selected_frame, :, :]
            input_dict[k] = selected_data
        return input_dict


class FuncWrapper(BaseTranformation):
    def __init__(self, keys: List[str], func: Callable, args: dict):
        self.func = func
        self.args = args
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, object]):
        for k in self.keys:
            input_dict[k] = self.func(input_dict[k], **self.args)
        return input_dict


class AxisFlip(BaseTranformation):
    def __init__(self, keys: List[str], axis, p: float = 0.5):
        assert (isinstance(axis, int) or isinstance(axis, tuple) or isinstance(axis, list))
        if not isinstance(p, float) or p < 0. or p > 1.:
            raise ValueError('p must be float between 0 and 1')
        self.axis = axis
        self.p = p
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        if random() < self.p:
            for k in self.keys:
                input_dict[k] = torch.flip(input_dict[k], dims=(self.axis,))
        return input_dict


class VerticalFlip(AxisFlip):
    def __init__(self, keys: List[str], p: float = 0.5):
        super().__init__(keys, -2, p)


class HorizontalFlip(AxisFlip):
    def __init__(self, keys: List[str], p: float = 0.5):
        super().__init__(keys, -1, p)
