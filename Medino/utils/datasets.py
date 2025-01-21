import os
import numpy as np
from torch.utils.data import Dataset
import fnmatch
import PIL.Image as Image


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            yield filename


class BaseFileListDataset(Dataset):
    def __init__(
            self,
            name_list,
            transform=None,
    ):
        super().__init__()
        self.name_list = [i.split('.')[0] for i in name_list]
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __filename__(self, idx):
        return self.name_list[idx]

    def __getitem__(self, idx):
        pass


class LUS_Dataset(BaseFileListDataset):
    def __init__(
            self,
            path_im,
            name_list,
            transform=None,
    ):
        super().__init__(name_list, transform)
        self.path_im = path_im
        self.name_list = self.reload_namelist()

    def reload_namelist(self):
        reloaded_namelist = []
        for name in self.name_list:
            for filename in find_files(self.path_im, f'*{name}*.npy'):
                reloaded_namelist.append(filename)
        return reloaded_namelist

    def __read_npy__(self, path_in, idx):
        return np.load(os.path.join(path_in, f"{self.name_list[idx]}"))

    def __getitem__(self, idx):
        input_dict = {'x': self.__read_npy__(self.path_im, idx),
                      # Read the data with shape (frames, channels, width, height)
                      'y': np.int64(combine_class(int(self.name_list[idx].split('_')[-1][0]))),
                      'names': self.name_list[idx]}
        if self.transform is not None:
            input_dict = self.transform(input_dict)
        return input_dict


class LUS_Videoset(BaseFileListDataset):
    def __init__(
            self,
            path_im,
            name_list,
            transform=None,
    ):
        super().__init__(name_list, transform)
        self.path_im = path_im
        self.name_list = self.reload_namelist()

    def reload_namelist(self):
        reloaded_namelist = []
        for name in self.name_list:
            for filename in find_files(self.path_im, f'*{name}*.npy'):
                reloaded_namelist.append(filename)
        return reloaded_namelist

    def __read_npy__(self, path_in, idx):
        return np.load(os.path.join(path_in, f"{self.name_list[idx]}"))

    def __getitem__(self, idx):
        input_dict = {
            'x': self.__read_npy__(self.path_im, idx),
            'y': np.int64(combine_class(int(self.name_list[idx].split('_')[-1][0]))),
            'names': self.name_list[idx]
        }
        if self.transform is not None:
            input_dict = self.transform(input_dict)
        return input_dict


class LUS_Imageset(BaseFileListDataset):
    def __init__(
            self,
            path_im,
            name_list,
            transform=None,
    ):
        super().__init__(name_list, transform)
        self.path_im = path_im
        self.name_list = self.reload_namelist()

    def reload_namelist(self):
        reloaded_namelist = []
        for name in self.name_list:
            for filename in find_files(self.path_im, f'*{name}*.png'):
                reloaded_namelist.append(filename)
        return reloaded_namelist

    def __read_png__(self, path_in, idx):
        image_path = os.path.join(path_in, f"{self.name_list[idx]}")
        image = Image.open(image_path)
        image = np.array(image)[np.newaxis, :]  # Convert to numpy array and add new axis
        image = image.astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1]
        return image

    def __getitem__(self, idx):
        input_dict = {
            'x': self.__read_png__(self.path_im, idx),
            'y': np.int64(combine_class(int(self.name_list[idx].split('_')[-1][0]))),
            'names': self.name_list[idx]
        }
        if self.transform is not None:
            input_dict = self.transform(input_dict)
        return input_dict


def combine_class(c):
    if c > 1:
        return c - 1
    else:
        return c


if __name__ == '__main__':
    DATA_PATH = f"./data/processed"
