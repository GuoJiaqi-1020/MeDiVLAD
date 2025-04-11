import glob
import os.path
import shutil
import time
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.nn import functional as F
from datetime import datetime
from typing import List, Dict
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_rootpath():
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return project_path


def low_disk_space(threshold_gb=10):
    """
    True if the free space is below the threshold, False otherwise.
    """
    current_path = os.getcwd()
    disk_usage = shutil.disk_usage(current_path)
    free_space_gb = disk_usage.free / (1024 ** 3)
    return free_space_gb < threshold_gb


class logger:
    def __init__(self, record: bool = True, **kwargs):
        """
        Logger for network training
        :param record: This should be a Boolean value, if False then network training will not be recorded
        """
        self.root = get_rootpath()
        if 'root' in kwargs.keys():
            self.root = kwargs['root']
        self.record = record
        time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        self.file = time_now  # Generate the folder name
        if 'folder_name' in kwargs.keys():
            if kwargs['folder_name'] is not None:
                self.file = kwargs['folder_name']
        self.trained_dir = os.path.join(self.root, 'trained_model', self.file)
        if self.record:
            self.mk_trained_dir()
        self.log_dir = f"{os.path.join(self.root, 'trained_model', self.file, f'log-{time_now}')}.txt"
        self.history = {}
        self.tajectory_index = {}
        self.log(content=f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        current_path = os.getcwd()
        disk_usage = shutil.disk_usage(current_path)
        free_space_gb = disk_usage.free / (1024 ** 3)
        self.log(content=f"Free space on disk: {free_space_gb} GB\n")
        if low_disk_space():
            self.log(content=f"Current trained_dir: {self.trained_dir}\n")

    def log(self, content: str):
        """Record the console output to a .txt file"""
        if self.record:
            with open(self.log_dir, 'a') as f:
                print(content)
                print(content, file=f)
        else:
            print(content)

    def mk_trained_dir(self):
        """Create a folder path ./root/trained_model/ to store the network model"""
        if not os.path.exists(os.path.join(self.root, 'trained_model')):
            os.makedirs(os.path.join(self.root, 'trained_model'))
        if not os.path.exists(self.trained_dir):
            os.makedirs(self.trained_dir)  # Name the folder according to the training time

    def log_file(self, suffix: str):
        """Generate the name for the stored file by indicating the suffix"""
        return os.path.join(self.trained_dir, self.file) + suffix


class Training_base(nn.Module):
    def __init__(self, record=True, root=None, **kwargs):
        super().__init__()
        self.root = root
        self.record = record
        self.folder_name = None
        if 'folder_name' in kwargs.keys():
            self.folder_name = kwargs['folder_name']
        self.Log = self.configure_logger()
        if 'args' in kwargs.keys():
            self.log_parameters(kwargs['args'])
        self.lr_scheduler = None
        self.optimizer = None
        self.model = None
        self.loss_fn = None
        self.num_classes = None
        self.batch_size = None
        self.trained_loader = None
        self.model = None

    def log_parameters(self, args):
        self.log(f"--------------Parameter Setting--------------\n"
                 f"{args}\n"
                 f"----------------------------------------------")

    def log(self, str_input: str):
        return self.Log.log(str_input)

    def forward(self, x):
        pX = self.model(x)
        return F.log_softmax(pX, dim=1)

    def configure_logger(self):
        return logger(
            root = self.root,
            record=self.record,
            folder_name=self.folder_name
        )

    def run_loss(self, batch_input: dict):
        return dict

    def run_metrics(self, batch_input: dict):
        return Tensor(), dict

    def train_model(self, loader):
        pass

    def test_model(self, loader):
        return

    def training_step(self, batch_input: dict, optimizer, **kwargs):
        self.model.train()
        optimizer.zero_grad()  # Reset gradients.
        loss, batch_output = self.run_loss(batch_input)
        batch_input.update(batch_output)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss, batch_input

    @torch.no_grad()
    def validation_step(self, batch_input: dict, **kwargs):
        # validation_step defines a single val loop.
        self.model.eval()
        loss, batch_output = self.run_loss(batch_input)
        batch_input.update(batch_output)
        return loss, batch_input

    def test_step(self, batch_input: dict):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch_input)

    def load(self, checkpoint_dir):
        if os.path.exists(checkpoint_dir):
            print('model loaded from %s' % checkpoint_dir)
            checkpoint = torch.load(checkpoint_dir)
            return self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self, Epoch: int, **kwargs):
        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict()}
        state.update(kwargs)
        torch.save(obj=state,
                   f=f"{self.Log.trained_dir}/ep_{Epoch}_acc_{self.history_best_acc * 100:.2f}.pth.tar")
        self.log(f"Checkpoint saved at epoch {Epoch}")
