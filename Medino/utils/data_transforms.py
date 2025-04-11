from typing import List, Dict, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from torch import from_numpy
import random
from random import randint
import torchvision.transforms.functional as transforms
import kornia.augmentation as K
from pytorchvideo.transforms.functional import uniform_temporal_subsample
import torch
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_plot(img1, img2, title='Demo'):
    plt.subplot(121)
    plt.title('Ori')
    plt.imshow(img1, cmap='gray', vmin=0, vmax=1)

    plt.subplot(122)
    plt.title('Changed')
    plt.imshow(img2, cmap='gray', vmin=0, vmax=1)

    plt.suptitle(f'{title}')
    plt.show()


def trans_none(input_dict: Dict[str, np.ndarray]):
    return input_dict


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


class CenterCrop(BaseTranformation):
    def __init__(self, keys: List[str], size_range: List[float], p: float):
        super().__init__(keys)
        self.size_range = size_range
        self.crop_ratio = None
        self.p = p

    def get_params(self, x: np.ndarray):
        h, w = x.shape[-2], x.shape[-1]
        th = round(x.shape[-2] * self.crop_ratio)
        tw = round(x.shape[-1] * self.crop_ratio)
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        i, j = (h - th) // 2, (w - tw) // 2
        return i, j, i + th, j + tw

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        self.crop_ratio = random.uniform(self.size_range[0], self.size_range[1])
        x = input_dict[self.keys[0]]
        i, j, h, w = self.get_params(x)
        centercrop = K.CenterCrop(size=(abs(h - i), abs(w - j)), p=0.5)
        for k in self.keys:
            cropped_img = centercrop(input_dict[k])
            input_dict[k] = cropped_img
        return input_dict


class RandomCrop(BaseTranformation):
    def __init__(self, keys: List[str], size_range: List[float], p: float):
        super().__init__(keys)
        self.size_range = size_range
        self.crop_ratio = None
        self.p = p

    def get_params(self, x: np.ndarray):
        ph = round(x.shape[-2] * 0.05)
        pw = round(x.shape[-1] * 0.05)
        th = round(x.shape[-2] * self.crop_ratio)
        tw = round(x.shape[-1] * self.crop_ratio)
        return th, tw, ph, pw

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        self.crop_ratio = random.uniform(self.size_range[0], self.size_range[1])
        x = input_dict[self.keys[0]]
        th, tw, ph, pw = self.get_params(x)
        randomcrop = K.RandomCrop3D(size=(x.shape[0], th, tw),
                                    padding=(pw, 0, pw, 0, 0, 0),
                                    p=self.p)
        for k in self.keys:
            cropped_img = randomcrop(input_dict[k].transpose(0, 1))
            input_dict[k] = cropped_img.squeeze(0).transpose(0, 1)
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
        if random.random() < self.p:
            for k in self.keys:
                input_dict[k] = torch.flip(input_dict[k], dims=(self.axis,))
        return input_dict


class VerticalFlip(AxisFlip):
    def __init__(self, keys: List[str], p: float = 0.5):
        super().__init__(keys, -2, p)


class HorizontalFlip(AxisFlip):
    def __init__(self, keys: List[str], p: float = 0.5):
        super().__init__(keys, -1, p)


class TimeFlip(AxisFlip):
    def __init__(self, keys: List[str], p: float = 0.5):
        super().__init__(keys, -4, p)


class RandomTransposeHW(BaseTranformation):
    def __init__(self, keys: List[str], p: float = 0.5):
        self.p = p
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        if random.random() < self.p:
            for k in self.keys:
                x_dims = list(range(len(input_dict[k].shape)))
                x_dims[-1] -= 1
                x_dims[-2] += 1

                input_dict[k] = input_dict[k].transpose(x_dims)

        return input_dict


class RandomRotate90(BaseTranformation):
    def __init__(self, keys: List[str], ks: int):
        self.ks = ks
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        ks = np.random.choice(self.ks)

        for k in self.keys:
            a1 = len(input_dict[k].shape) - 2
            a2 = len(input_dict[k].shape) - 1
            input_dict[k] = np.rot90(input_dict[k], k=ks, axes=(a1, a2))

        return input_dict


class ColorJiggle(BaseTranformation):
    def __init__(self, keys: List[str], p: float, intensity: float):
        self.transform = K.ColorJiggle(
            brightness=intensity,
            contrast=intensity,
            p=p,
            same_on_batch=True)
        # the input is (batch, channel, H, W) which is a single video -> perform same_on_batch = True
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        for k in self.keys:
            input_dict[k] = self.transform(input_dict[k])
        return input_dict


class CenterRescale(BaseTranformation):
    def __init__(self, p, rescale_factor, keys: List[str]):
        self.ky = False
        dice = random.random()
        if dice <= p:
            self.ky = True
        self.rescale_factor = rescale_factor
        super().__init__(keys)

    def get_params(self, x: np.ndarray):
        h, w = x.shape[-2], x.shape[-1]
        th, tw = int(h * self.enlarge), int(w * self.enlarge),
        i, j = (h - th) // 2, (w - tw) // 2
        if i + th > x.shape[-2] or j + tw > x.shape[-1]:
            return i, j, x.shape[-2], x.shape[-1]
        return i, j, i + th, j + tw

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        self.enlarge = 1 - np.random.choice(self.rescale_factor) / 100
        x = input_dict[self.keys[0]]
        i, j, h, w = self.get_params(x)
        if self.ky:
            for k in self.keys:
                new_img = np.squeeze(input_dict[k][..., i:h, j:w], 0)
                input_dict[k] = cv2.resize(new_img, x.shape[1:], interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
        else:
            pass
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
        output_dict = {}
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


class VisualLayer(BaseTranformation):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        for k in self.keys:
            if k == 'y':
                image_plot(np.squeeze(input_dict['y_ori'], 0), np.squeeze(input_dict[k], 0), title='Confocal')
            else:
                image_plot(np.squeeze(input_dict['x_ori'], 0), np.squeeze(input_dict[k], 0), title='PWS')
        return input_dict


class PaddVideo(BaseTranformation):
    def __init__(self, keys: List[str], num_frames: int, random: bool = False):
        super().__init__(keys)
        self.frames_num = num_frames
        self.random = random

    @staticmethod
    def random_uniform_sample(x: torch.Tensor, num_samples: int) -> torch.Tensor:
        num_frames = x.shape[0]
        sample_interval = int(num_frames // num_samples)

        assert sample_interval >= 1
        try:
            start_offset = torch.randint(0, num_frames - sample_interval * num_samples, (1,)).item()
        except:
            start_offset = 0
        indices = torch.arange(start_offset, num_frames, step=sample_interval)[:num_samples]
        sampled_video = x[indices]
        return sampled_video

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        for k in self.keys:
            x = input_dict[k]
            if x.shape[0] <= self.frames_num or not self.random:
                input_dict[k] = uniform_temporal_subsample(
                    x=x,
                    num_samples=self.frames_num,
                    temporal_dim=0)
            else:
                assert x.shape[0] > self.frames_num
                input_dict[k] = self.random_uniform_sample(
                    x=x,
                    num_samples=self.frames_num
                )
        return input_dict


class Resize(BaseTranformation):
    def __init__(self, keys: List[str], size: List[int]):
        super().__init__(keys)
        self.size = size

    def resize_video(self, x):
        # Create transform to resize each frame
        resized_frame = transforms.resize(x, size=self.size, antialias=True)
        resized_frame = resized_frame.repeat(1, 3, 1, 1)
        return resized_frame

    def __call__(self, input_dict: Dict[str, torch.tensor]):
        for k in self.keys:
            resized = self.resize_video(x=input_dict[k])
            input_dict[k] = resized
        return input_dict


class CutVideo(BaseTranformation):
    def __init__(self, keys: List[str], num_frames: int):
        super().__init__(keys)
        self.frames_num = num_frames

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        for k in self.keys:
            input_dict[k] = uniform_temporal_subsample(x=input_dict[k],
                                                       num_samples=self.frames_num,
                                                       temporal_dim=0)
        return input_dict


class To_Score(BaseTranformation):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, torch.Tensor]):
        for k in self.keys:
            input_dict[k] = input_dict[k].to(torch.float32)
        return input_dict
