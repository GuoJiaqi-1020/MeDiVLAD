import argparse
import csv
import os
import random
import sys
import time
import warnings
import numpy as np

exp_root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Train a DINO model.")
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate for the optimizer.')
parser.add_argument('--iters', type=int, default=50, help='Iteration steps')
parser.add_argument('--epochs', type=int, default=200, help='Total number of epochs for training.')
parser.add_argument('--CUDA', type=str, nargs='+', default=[0], help='Indices of GPUs to use (e.g., 0 1 2).')
parser.add_argument('--warm_up', type=int, default=500, help='Number of samples per batch.')
parser.add_argument('--batch_size', type=int, default=2, help='Number of samples per batch.')
parser.add_argument('--config', type=str, default='validate_dinosmall_videocls')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.CUDA))

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, f1_score
from Medino.dino_utils.dino_backbone import DINO
from Medino.dino_utils.dino_augmentation import KNN_augmentation
from Medino.dino_utils.basic_transform import DinoAug
from Medino.dino_utils.utils import cosine_scheduler
from Medino.models.medivlad import MediVLAD
from Medino.utils.datasets import LUS_Videoset
from Medino.utils.train_wrapper import Training_base
from Medino.utils.basic_utils import check_lus_data, load_yaml, add_dict_to_argparser
from Medino.utils.data_transforms import (Resize, HorizontalFlip, TimeFlip, Compose,
                                          ToTensor, PaddVideo, CenterCrop, ColorJiggle)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_parts(filename):
    parts = filename.split('_')
    if len(parts) >= 5:
        return parts[2], '_'.join(parts[3:])
    else:
        return None, None


def seed_everything(seed=42):
    """Function to set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def choose_classifier():
    if args.model_type == 'base':
        return nn.Linear(768, 3).to(device)
    elif args.model_type == 'small':
        return nn.Linear(384, 3).to(device)
    elif args.model_type == 'tiny':
        return nn.Linear(192, 3).to(device)
    elif args.model_type == 'vit_tinyer':
        return nn.Linear(96, 3).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')


class MediVLAD_Train(Training_base):
    def __init__(self, args=args, record: bool = True, folder_name: str = None):
        super().__init__(record=record, args=args, folder_name=folder_name, root=exp_root)
        self.best_top1_acc = None
        self.args = args
        self.dropout = self.args.dropout

        backbone, linear = self.select_modules()

        self.model = MediVLAD(
            embedding_backbone=backbone[0],
            num_classes=3,
            num_clusters=3,
            warm_up=args.warm_up,
            seq_len=args.seq_len,
            dropout=args.dropout,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            vlad_centroid=args.vlad_centroid,
            num_layers=1,
        ).to(device)

        self.model = nn.DataParallel(self.model)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.wd)

        self.load_model_checkpoint(os.path.join('dino_checkpoint/video_cls/', self.args.checkpoint))

        self.lr_scheduler = cosine_scheduler(
            base_value=args.lr,
            final_value=5e-5,
            epochs=args.epochs,
            niter_per_ep=args.iters,
            warmup_epochs=0
        )

        self.cross_entropy = CrossEntropyLoss()

    def load_model_checkpoint(self, checkpoint_path: str):
        """Loads model and optimizer states from a checkpoint."""
        try:
            checkpoint = torch.load(f'{root}/{checkpoint_path}', map_location=device)
            self.log(f'\n### State dict found, Loading...')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            self.log(f'\n### Loaded checkpoint: {checkpoint_path} ###\n')
        except:
            self.log(f'\n### No checkpoint found at {checkpoint_path}, train from scratch ###\n')

    def load_pretrained_dinobackbone(self, checkpoint: str, dino_backbone):
        self.log(f'\n#### Loading pre-trained BACKBONE ####')
        flag = False
        try:
            state_dict = torch.load(f'{root}/dino_checkpoint/backbone/{checkpoint}.pth')
            if 'backbone' in state_dict:
                state_dict = state_dict['backbone']
            try:
                self.log(f'# DINO IMG BACKBONE -> Try to Load: {checkpoint}')
                dino_backbone.load_state_dict(state_dict)
                flag = True
            except Exception as e:
                self.log(f'# DINO IMG BACKBONE -> Try to Load: {checkpoint} AGAIN')
                dino_backbone[0].load_state_dict(state_dict)
                flag = True
        except Exception as final_e:
            self.classifier_flag = False
            self.log(f'# Error loading {checkpoint} *** ')
            self.log('# Randomly initializing DINO IMG model ***')
        if flag:
            self.log('Pretrained BACKBONE loaded...')
        if self.args.backbone_type == 'resnet':
            return dino_backbone
        return dino_backbone[0]

    def select_modules(self):
        if self.args.backbone_type == 'dino':
            backbone = DINO(
                model_type=self.args.model_type,
                out_dim=self.args.dino_out_dim, use_bn=False).teacher.to(device)
            self.log('Loading pretrained DINO!\n')
            linear = torch.nn.Sequential(
                choose_classifier(),
            ).to(device)
        else:
            raise ValueError
        return backbone, linear

    def train_classifier(self, train_loader, val_loader, epochs):
        """Training loop for the DINO model with linear classifier."""
        _, overall_accuracy, weighted_accuracy = self.validate(val_loader)
        best_accuracy = overall_accuracy + weighted_accuracy
        self.log(f'Best accuracy initialized at: {overall_accuracy:.4f}')
        for epoch in range(epochs):
            self.log(f'\n********************# {epoch + 1} #**********************')
            running_loss = 0.0
            all_preds = []
            all_labels = []
            self.model.train()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{epochs}')
            for it, batch_input in progress_bar:
                it = it + epoch * args.iters
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = self.lr_scheduler[it]

                for key, value in batch_input.items():
                    if isinstance(value, torch.Tensor):
                        batch_input[key] = value.to(device)

                # Forward pass through the linear classifier
                outputs = self.model(batch_input['x'], ep=epoch + 1, dropout=self.dropout)

                # Compute loss
                loss = self.cross_entropy(outputs, batch_input['y'])
                progress_bar.set_postfix(loss=loss.item())
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate running loss for logging
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_input['y'].cpu().numpy())

            self.log('Learning rate: {:.3e}'.format(self.optimizer.param_groups[0]['lr']))
            self.log('Dropout rate: {:.3f}'.format(self.dropout))
            # Log the average loss for the epoch
            avg_loss = running_loss / len(train_loader)

            # Calculate overall accuracy
            train_accuracy = accuracy_score(all_labels, all_preds)
            val_loss, overall_accuracy, weighted_accuracy = self.validate(val_loader)

            # Register the results
            self.log(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f} '
                f'-> Overall Accuracy: {train_accuracy:.4f}')
            self.log(
                f'Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.4f}, '
                f'Overall Accuracy: {overall_accuracy:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}\n'
            )

            # Save the model checkpoint if overall accuracy improves (based on both overall/weighted acc)
            if overall_accuracy + weighted_accuracy > best_accuracy:
                best_accuracy = overall_accuracy + weighted_accuracy
                checkpoint_path = (f'{self.Log.trained_dir}/VAL_F{self.args.fold}_{self.args.backbone_type}'
                                   f'_acc_{overall_accuracy * 10000:.0f}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                self.log(f'Saved checkpoint: {checkpoint_path}')

    @torch.no_grad()
    def validate(self, val_loader, num_classes=3):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_input in tqdm(val_loader):
                for key, value in batch_input.items():
                    if isinstance(value, torch.Tensor):
                        batch_input[key] = value.to(device)

                # Forward pass through the model
                outputs = self.model(batch_input['x'])

                # Option to process max probability
                outputs = torch.nn.functional.softmax(outputs, dim=1)

                # Compute loss
                loss = self.cross_entropy(outputs, batch_input['y'])
                val_loss += loss.item()

                # Store predictions, probabilities, and labels
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(outputs.cpu().numpy())  # Store probabilities for ROC-AUC
                all_labels.extend(batch_input['y'].cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_labels, all_preds)

        # Calculate weighted accuracy (weighted F1 score)
        weighted_accuracy = f1_score(all_labels, all_preds, average='weighted')

        # Binarize labels for multi-class ROC-AUC calculation
        all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])

        # Calculate ROC-AUC (one-vs-all) for each class
        roc_auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr')

        self.log(f'ROC-AUC (One-vs-All): {roc_auc:.4f}')

        return avg_val_loss, overall_accuracy, weighted_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    add_dict_to_argparser(parser, default_dict={**load_yaml(f'{exp_root}/configs/video/{args.config}.yaml')})
    args = parser.parse_args()

    ################################ Load the dataset ###################################
    # TODO: Replace DATA_PATH with the path to the dataset.
    # TODO:
    #   - val_list: List of video names used for validation.
    #   - train_list: List of video names used for training.
    # TODO: Download the dataset from the link provided in the README.md.

    #####################################################################################
    DATA_PATH = f"{root}/data/15_frames_processed/"

    with open(f'{root}/data/2_fold_validation/train{args.fold}.csv', 'r') as f:
        reader = csv.reader(f)
        train_list = list(set([item[0].rsplit('_index', 1)[0] for item in reader]))
    with open(f'{root}/data/2_fold_validation/validation{args.fold}.csv', 'r') as f:
        reader = csv.reader(f)
        val_list = list(set([item[0].rsplit('_index', 1)[0] for item in reader]))
    #####################################################################################

    # Read the configuration file and name the log folder
    config = extract_parts(args.config)
    out_name = (f"VIDEO_{args.backbone_type}{args.model_type}-{config[0]}-{config[1]}-"
                f"-{time.strftime('%m%d_%H%M_%S', time.localtime())}")

    Medino = MediVLAD_Train(
        args=args,
        folder_name=out_name,
        record=True
    )

    Medino.log(f'\nMEAN and STD for Normalization: {args.mean_std}')

    supervised_aug = KNN_augmentation(
        image_size=224,
        mean_std=args.mean_std,
    )

    # Define the dataset transformation
    train_transform = Compose([
        ToTensor(keys=['x'], contiguous=True),
        HorizontalFlip(keys=['x'], p=0.5),
        TimeFlip(keys=['x'], p=0.5),
        ColorJiggle(keys=['x'], p=1.0, intensity=0.4),
        CenterCrop(keys=['x'], size_range=[0.4, 1], p=0.5),
        Resize(keys=['x'], size=[224, 224]),
        PaddVideo(keys=['x'], num_frames=args.seq_len, random=True),
        DinoAug(keys=['x'], transform=supervised_aug),
    ])

    val_transform = Compose([
        ToTensor(keys=['x'], contiguous=True),
        Resize(keys=['x'], size=[224, 224]),
        PaddVideo(keys=['x'], num_frames=args.seq_len, random=False),
        DinoAug(keys=['x'], transform=supervised_aug),
    ])

    training_set = LUS_Videoset(DATA_PATH, train_list, transform=train_transform)
    val_set = LUS_Videoset(DATA_PATH, val_list, transform=val_transform)
    vlad_set = LUS_Videoset(DATA_PATH, train_list, transform=val_transform)

    Medino.log(f'Length of validation set: {len(val_set)}')
    Medino.log(f'Length of training set: {len(training_set)}')

    train_sampler = WeightedRandomSampler(
        torch.ones(len(training_set)).double(),
        num_samples=args.batch_size * args.iters
    )
    TrainLoader = DataLoader(
        training_set, sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=12, pin_memory=True)
    ValLoader = DataLoader(
        val_set, batch_size=2, drop_last=True,
        num_workers=12, pin_memory=True)
    VladLoader = DataLoader(
        vlad_set, batch_size=1, drop_last=True,
        num_workers=12, pin_memory=True)

    Medino.train_classifier(
        train_loader=TrainLoader,
        val_loader=ValLoader,
        epochs=args.epochs,
    )
