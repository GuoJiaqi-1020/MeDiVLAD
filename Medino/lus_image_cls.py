import argparse
import csv
import os
import random
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
exp_root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root)

parser = argparse.ArgumentParser(description="Validate a DINO model.")
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer.')
parser.add_argument('--iters', type=int, default=100, help='Total number of iterations for training.')
parser.add_argument('--epochs', type=int, default=250, help='Total number of epochs for training.')
parser.add_argument('--CUDA', type=int, nargs='+', default=[0], help='Indices of GPUs to use (e.g., 0 1 2).')
parser.add_argument('--batch_size', type=int, default=2, help='Number of samples per batch.')
parser.add_argument('--config', type=str, default='validate_dinosmall_imagecls')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.CUDA))

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Linear
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from Medino.utils.datasets import LUS_Imageset
from Medino.utils.train_wrapper import Training_base
from Medino.utils.basic_utils import check_lus_data, load_yaml, add_dict_to_argparser
from Medino.dino_utils.dino_backbone import DINO, DownStream
from Medino.dino_utils.utils import cosine_scheduler
from Medino.dino_utils.knn_utils import extract_features, knn_classifier
from Medino.dino_utils.dino_augmentation import KNN_augmentation
from Medino.dino_utils.basic_transform import Compose, ToTensor, Resize, HorizontalFlip, DinoAug

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
        return Linear(768, 3).to(device)
    elif args.model_type == 'small':
        return Linear(384, 3).to(device)
    elif args.model_type == 'tiny':
        return Linear(192, 3).to(device)
    elif args.model_type == 'vit_tinyer':
        return Linear(96, 3).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')


class Medino_train(Training_base):
    def __init__(self, args=args, record: bool = True, folder_name: str = None):
        super().__init__(record=record, args=args, folder_name=folder_name, root=exp_root)
        self.args = args

        backbone, linear = self.select_modules()

        self.model = DownStream(
            backbone_type=args.backbone_type,
            backbone=backbone,
            classifier=linear,
            checkpoint=args.checkpoint,
            train_classifier=args.train_classifier,
            train_backbone=args.train_backbone,
            logger=self.Log,
            device=device,
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.wd)

        self.lr_scheduler = cosine_scheduler(
            base_value=args.lr,
            final_value=1e-6,
            epochs=args.epochs,
            niter_per_ep=args.iters,
            warmup_epochs=0
        )

        self.cross_entropy = CrossEntropyLoss()

    def knn_validation(self, train_loader, val_loader, k_values, T, num_classes=3):
        """Training loop for the DINO model with linear classifier."""
        feature_extractor = self.model.backbone

        # Extract features and labels from training and validation data
        train_features, train_labels = extract_features(train_loader, feature_extractor, device)
        test_features, test_labels = extract_features(val_loader, feature_extractor, device)
        vlad_features = torch.nn.functional.normalize(train_features, p=2, dim=1)

        # Initialize a dictionary to hold the sum of features and count for each class
        class_feature_sums = {i: torch.zeros_like(train_features[0]) for i in range(num_classes)}
        class_counts = {i: 0 for i in range(num_classes)}

        # Calculate sum of features for each class
        for feature, label in zip(vlad_features, train_labels):
            class_feature_sums[label.item()] += feature
            class_counts[label.item()] += 1

        # Calculate mean features for each class
        # (This is for preparing for subsequent video-level classification; not mandatory and can be commented out.)
        ##############################################################
        # class_feature_means = []
        # for i in range(num_classes):
        #     if class_counts[i] > 0:
        #         class_feature_means.append(class_feature_sums[i] / class_counts[i])
        #     else:
        #         class_feature_means.append(torch.zeros_like(vlad_features[0]))
        # class_feature_means_tensor = torch.stack(class_feature_means, dim=0)
        # os.makedirs('./centroid', exist_ok=True)
        # torch.save(class_feature_means_tensor, f'./centroid/centroid_{self.args.checkpoint}.pt')
        ################################################################

        results = []
        for k in k_values:
            top1, top2 = knn_classifier(
                train_features, train_labels,
                test_features, test_labels,
                k, T, num_classes)
            results.append((k, top1, top2))

        max_top1 = max(results, key=lambda item: item[1])
        max_top2 = max(results, key=lambda item: item[2])
        self.log(f"\nMax Top1 Acc {max_top1[1]} at k = {max_top1[0]}")
        return max_top1, max_top2

    def select_modules(self):
        if self.args.backbone_type == 'dino':
            backbone = DINO(
                model_type=self.args.model_type,
                out_dim=self.args.dino_out_dim, use_bn=False).teacher.to(device)
            self.log('Training pretrained DINO!\n')
            linear = torch.nn.Sequential(
                choose_classifier(),
            ).to(device)
        else:
            raise ValueError
        return backbone, linear

    def train_classifier(self, knn_train_loader, train_loader, val_loader, epochs):
        """Training loop for the DINO model with linear classifier."""
        self.log('\n################ Initial KNN Validation ################')
        initial_top1_acc, initial_top2_acc = self.knn_validation(
            knn_train_loader, val_loader, k_values=range(1, 20), T=0.07
        )
        self.log(f'Initial Top1 accuracy: {initial_top1_acc[1]:.4f}')

        val_loss, overall_accuracy, weighted_accuracy = self.validate(val_loader)
        best_accuracy = overall_accuracy  # Initialize best accuracy
        self.log(f'\n###########'
                 f'Initial LINEAR ACC: {best_accuracy:.4f}'
                 f'###########\n')
        assert self.args.train_classifier or self.args.train_backbone, 'Train classifier and backbone are not enabled!'

        for epoch in range(epochs):
            self.log(f'**************************# {epoch} #***************************')
            running_loss = 0.0
            self.model.train()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{epochs}')
            for i, batch_input in progress_bar:
                i = i + epoch * args.iters
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr_scheduler[i]

                for key, value in batch_input.items():
                    if isinstance(value, torch.Tensor):
                        batch_input[key] = value.to(device)

                # Forward pass through the linear classifier
                outputs = self.model(batch_input['x'])

                # Compute loss
                loss = self.cross_entropy(outputs, batch_input['y'])
                progress_bar.set_postfix(loss=loss.item())
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Accumulate running loss for logging
                running_loss += loss.item()

            self.log('Learning rate: {:.3e}'.format(self.optimizer.param_groups[0]['lr']))
            # Log the average loss for the epoch
            avg_loss = running_loss / len(train_loader)

            # Validate the model
            val_loss, overall_accuracy, weighted_accuracy = self.validate(val_loader)

            self.log(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}')
            self.log(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.4f}, '
                     f'Overall Accuracy: {overall_accuracy:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}\n'
                     )

            # Save the model checkpoint if overall accuracy improves
            if overall_accuracy > best_accuracy:
                best_accuracy = overall_accuracy
                checkpoint_path = (f'{self.Log.trained_dir}/VAL_F{self.args.fold}_{self.args.backbone_type}'
                                   f'_acc_{overall_accuracy * 10000:.0f}.pth')
                torch.save({
                    'backbone': self.model.backbone.module.state_dict(),
                    'linear_state_dict': self.model.classifier.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                self.log(f'Saved checkpoint: {checkpoint_path}')

    @torch.no_grad()
    def validate(self, val_loader):
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

                # Forward pass through the linear classifier
                outputs = self.model(batch_input['x'])
                # heat_map = self.model.backbone.module.get_last_selfattention(batch_input['x'])
                prob = torch.nn.functional.softmax(outputs, dim=1)

                # Compute loss
                loss = self.cross_entropy(prob, batch_input['y'])
                val_loss += loss.item()

                # Store predictions and labels for accuracy calculation
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(prob.cpu().numpy())
                all_labels.extend(batch_input['y'].cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_labels, all_preds)

        all_labels_bin = label_binarize(all_labels, classes=[i for i in range(3)])

        # Calculate one-vs-all ROC-AUC for each class
        roc_auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr')

        self.log(f"ROC-AUC (One-vs-All): {roc_auc}\n"
                 f"#################################")

        # Calculate weighted accuracy (weighted F1 score)
        weighted_accuracy = f1_score(all_labels, all_preds, average='weighted')

        return avg_val_loss, overall_accuracy, weighted_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    add_dict_to_argparser(parser, default_dict={**load_yaml(f'{exp_root}/configs/image/{args.config}.yaml')})
    args = parser.parse_args()
    config = extract_parts(args.config)

    out_name = (f"F{args.fold}_VALID_{args.backbone_type}{args.model_type}-{config[1]}-"
                f"{args.checkpoint}-{time.strftime('%m%d_%H%M_%S', time.localtime())}")
    Medino = Medino_train(args=args, folder_name=out_name, record=True)

    Medino.log(f'\n################ CONFIG ################'
               f'\nMEAN and STD for Normalization: {args.mean_std}')
    supervised_train_aug = KNN_augmentation(
        global_crops_scale=args.sup_train_global_crops_scale,
        image_size=224,
        mean_std=args.mean_std,
    )

    supervised_val_aug = KNN_augmentation(
        image_size=224,
        mean_std=args.mean_std,
    )

    train_transform = Compose([
        ToTensor(keys=['x'], contiguous=True),
        HorizontalFlip(keys=['x'], p=0.5),
        Resize(keys=['x'], size=(224, 224)),
        DinoAug(keys=['x'], transform=supervised_train_aug),
    ])

    val_transform = Compose([
        ToTensor(keys=['x'], contiguous=True),
        Resize(keys=['x'], size=(224, 224)),
        DinoAug(keys=['x'], transform=supervised_val_aug),
    ])

    ################################ Load the dataset ###################################
    # TODO: Replace DATA_PATH with the path to the dataset
    # TODO: val_list & train_list are the list of labeled images for validation and training
    # TODO: Download the dataset from the link provided in the README.md
    #####################################################################################
    DATA_PATH = f"{root}/data/labeled_frames/"

    with open(f'{root}/data/2_fold_validation/train{args.fold}.csv', 'r') as f:
        reader = csv.reader(f)
        train_list = [item[0] for item in reader]
    with open(f'{root}/data/2_fold_validation/validation{args.fold}.csv', 'r') as f:
        reader = csv.reader(f)
        val_list = [item[0] for item in reader]
    #####################################################################################

    training_set = LUS_Imageset(DATA_PATH, train_list, transform=train_transform)
    knn_train_set = LUS_Imageset(DATA_PATH, train_list, transform=val_transform)
    val_set = LUS_Imageset(DATA_PATH, val_list, transform=val_transform)

    Medino.log(f'Length of validation set: {len(val_set)}')
    Medino.log(f'Length of training set: {len(training_set)}')

    TrainLoader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        num_workers=12, shuffle=True, pin_memory=True)

    KNN_TrainLoader = DataLoader(
        knn_train_set,
        batch_size=1,
        drop_last=True,
        num_workers=12,
        pin_memory=True)
    ValLoader = DataLoader(
        val_set,
        batch_size=1,
        drop_last=True,
        num_workers=12,
        pin_memory=True)

    Medino.train_classifier(
        train_loader=TrainLoader,
        val_loader=ValLoader,
        knn_train_loader=KNN_TrainLoader,
        epochs=args.epochs,
    )
