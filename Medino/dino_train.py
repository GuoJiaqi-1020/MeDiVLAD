import argparse
import csv
import os
import random
import sys
import time
import numpy as np

exp_root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root)
parser = argparse.ArgumentParser(description="Train a DINO model.")

parser.add_argument('--iters', type=int, default=100, help='Number of iterations to run.')
parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs for training.')
parser.add_argument('--CUDA', type=int, nargs='+', default=[0], help='Indices of GPUs to use.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--note', type=str, default='LUS', help='Additional note or tag for the experiment.')
parser.add_argument('--config', type=str, default='dino_train')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.CUDA))

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from Medino.utils.datasets import LUS_Dataset, LUS_Imageset
from Medino.utils.train_wrapper import Training_base
from Medino.utils.basic_utils import check_lus_data, load_yaml, add_dict_to_argparser
from Medino.dino_utils.dino_backbone import DINO, DINOLoss
from Medino.dino_utils.knn_utils import knn_classifier, extract_features
from Medino.dino_utils.basic_transform import Compose, ToTensor, RandomSelect, DinoAug, Resize
from Medino.dino_utils.utils import cancel_gradients_last_layer, cosine_scheduler, get_params_groups, clip_gradients
from Medino.dino_utils.dino_augmentation import DataAugmentationDINO, KNN_augmentation


def seed_everything(seed=42):
    """Function to set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Medino_train(Training_base):
    def __init__(self, args=args, record: bool = True, folder_name: str = None, device=None):
        super().__init__(record=record, args=args, folder_name=folder_name, root=exp_root)
        self.args = args
        self.best_top1_acc = 0
        self.device = device
        self.dino = DINO(
            model_type=args.model_type,
            out_dim=args.dino_out_dim,
            use_bn=False).to(self.device)

        if args.load_pretrained_backbone:
            self.init_model_weights(model_type='student')
            self.init_model_weights(model_type='teacher')
        for p_s in self.dino.student.module.parameters():
            p_s.requires_grad = True
        for p_t in self.dino.teacher.parameters():
            p_t.requires_grad = False

        self.loss = DINOLoss(
            out_dim=args.dino_out_dim, ncrops=(2 + args.local_crops_number), student_temp=0.1,
            warmup_teacher_temp=args.warmup_teacher_temp, teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs, nepochs=args.epochs)

        self.lr_scheduler = cosine_scheduler(
            base_value=5e-4 * (args.batch_size / 256),
            final_value=1e-6,
            epochs=args.epochs,
            niter_per_ep=args.iters,
            warmup_epochs=10
        )

        self.wd_scheduler = cosine_scheduler(
            base_value=0.1,
            final_value=0.5,
            epochs=args.epochs,
            niter_per_ep=args.iters,
        )

        self.momentum_scheduler = cosine_scheduler(
            base_value=args.init_momentum,
            final_value=1,
            epochs=args.epochs,
            niter_per_ep=args.iters,
        )

        params_groups = get_params_groups(self.dino.student.module)
        self.optimizer = AdamW(params_groups)

    def init_model_weights(self, model_type: str):
        state_dict = torch.load(f'{root}/dino_checkpoint/backbone/{args.checkpoint}.pth')
        backbone_state_dict = {}
        head_state_dict = {}
        if model_type in state_dict.keys():
            for key, value in state_dict[model_type].items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                    backbone_state_dict[new_key] = value
                elif key.startswith('head.'):
                    new_key = key.replace('head.', '')
                    head_state_dict[new_key] = value
                elif key.startswith('module.backbone.'):
                    new_key = key.replace('module.backbone.', '')
                    backbone_state_dict[new_key] = value
                elif key.startswith('module.head.'):
                    new_key = key.replace('module.head.', '')
                    head_state_dict[new_key] = value
                else:
                    raise Exception
        else:
            backbone_state_dict = state_dict
            self.log(f'Load statedict without model type: {model_type}')
            self.dino.teacher.load_state_dict(backbone_state_dict, strict=True)
            return None

        self.log(f'Loading pre-trained model: {args.checkpoint}')
        if model_type == 'teacher':
            self.dino.teacher[0].load_state_dict(backbone_state_dict, strict=True)
            self.dino.teacher[1].load_state_dict(head_state_dict, strict=True)
        elif model_type == 'student':
            self.dino.student.module[0].load_state_dict(backbone_state_dict, strict=True)
            self.dino.student.module[1].load_state_dict(head_state_dict, strict=True)
        else:
            raise Exception

    def make_augmentation(self, local_crops_number) -> dict:
        """Function to create the dataset for training and validation."""
        data_aug = {}
        self.log(f'*** Augmentation Details ***')
        global_crops_scale = (0.6, 1.0)
        local_crops_scale = (0.1, 0.6)
        self.log(f'GCS: {global_crops_scale}')
        self.log(f'LCS: {local_crops_scale}')

        dino_aug = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            image_size=224,
        )

        mean_std = ([0.3261, 0.3261, 0.3261],
                    [0.2283, 0.2283, 0.2283])
        # mean and std of the entire LUS dataset

        self.log(f'MEAN and STD for Norm: {mean_std}')
        knn_aug = KNN_augmentation(
            image_size=224,
            mean_std=mean_std
        )  # Only apply normalization

        data_aug.update(
            {'train_transform': Compose(
                [ToTensor(keys=['x'], contiguous=True),
                 RandomSelect(keys=['x']),
                 Resize(keys=['x'], size=(224, 224)),
                 DinoAug(keys=['x'], transform=dino_aug),
                 ])})

        data_aug.update(
            {'knn_transform': Compose([
                ToTensor(keys=['x'], contiguous=True),
                Resize(keys=['x'], size=(224, 224)),
                DinoAug(keys=['x'], transform=knn_aug),
            ])})

        self.log(f"Augmentation pipeline created:\n"
                 f"{data_aug}\n")
        return data_aug

    def make_dataset(self, data_path, dinoaug, fold) -> dict:
        """Function to create the dataset for training and validation."""
        self.log(f'*** Dataset Details ***')
        dataset_dict = {}
        with open(f'{root}/data/video_list.csv', 'r') as f:
            reader = csv.reader(f)
            dino_train_list = [item[0] for item in reader]
        with open(f'{root}/data/2_fold_validation/train{fold}.csv', 'r') as f:
            reader = csv.reader(f)
            knn_train = [item[0] for item in reader]
        with open(f'{root}/data/2_fold_validation/validation{fold}.csv', 'r') as f:
            reader = csv.reader(f)
            knn_val = [item[0] for item in reader]

        # Prepare dataset, for dino training and KNN validation
        training_set = LUS_Dataset(data_path[0], dino_train_list, transform=dinoaug['train_transform'])
        knn_trainset = LUS_Imageset(data_path[1], knn_train, transform=dinoaug['knn_transform'])
        knn_valset = LUS_Imageset(data_path[1], knn_val, transform=dinoaug['knn_transform'])
        self.log(
            f'Length of Dataset:\n'
            f'DINO_train: {len(training_set)}\n'
            f'KNN_train: {len(knn_trainset)}\n'
            f'KNN_val: {len(knn_valset)}\n'
        )

        # Create dataloader for training and validation
        train_sampler = WeightedRandomSampler(
            torch.ones(len(training_set)).double(),
            num_samples=args.batch_size * args.iters)

        dataset_dict.update(
            {
                'train_loader': DataLoader(
                    training_set, batch_size=args.batch_size, num_workers=12,
                    sampler=train_sampler, pin_memory=True, drop_last=True),
                'knn_train': DataLoader(
                    knn_trainset, batch_size=1, num_workers=12, pin_memory=True),
                'knn_val': DataLoader(
                    knn_valset, batch_size=1, num_workers=12, pin_memory=True)}
        )

        return dataset_dict

    def training_step(self, train_loader, epoch, **kwargs):
        """Train the model for one epoch."""
        self.dino.student.module.train()
        freeze_last_layer = 1

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
        accumulated_loss = 0.0
        num_batches = len(train_loader)
        for it, input_dict in progress_bar:
            student_outputs, teacher_outputs = [], []
            it = args.iters * epoch + it
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_scheduler[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_scheduler[it]

            for j, image in enumerate(input_dict['x'][1]):
                image = image.to(self.device)
                student_output = self.dino(image)
                student_outputs.append(student_output)

                if j < 2:
                    with torch.no_grad():
                        teacher_output = self.dino(image, is_teacher=True)
                        teacher_outputs.append(teacher_output)

            student_outputs_tensor = torch.stack(student_outputs)
            teacher_outputs_tensor = torch.stack(teacher_outputs)
            loss = self.loss(
                student_output=student_outputs_tensor,
                teacher_output=teacher_outputs_tensor,
                epoch=epoch
            )

            self.optimizer.zero_grad()

            loss.backward()
            clip_gradients(self.dino.student.module, 3.0)
            cancel_gradients_last_layer(epoch, self.dino.student.module, freeze_last_layer)
            self.optimizer.step()

            accumulated_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            with torch.no_grad():
                m = self.momentum_scheduler[it]
                for param_student, param_teacher in zip(self.dino.student.module.parameters(),
                                                        self.dino.teacher.parameters()):
                    param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

        average_loss = accumulated_loss / num_batches
        self.log(f'Weight Decay: {self.wd_scheduler[it]:.3e}; '
                 f'Current lr: {self.lr_scheduler[it]:.3e}; Momentum: {m:.3e}')
        return average_loss

    def knn_validation(self, train_loader, val_loader, k_values, T, num_classes=3):
        """Training loop for the DINO model with linear classifier."""
        self.dino.teacher.eval()
        feature_extractor = self.dino.teacher[0].to(self.device)

        # Extract features and labels from training and validation data
        train_features, train_labels = extract_features(train_loader, feature_extractor, self.device)
        test_features, test_labels = extract_features(val_loader, feature_extractor, self.device)

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

    @torch.no_grad()
    def validation_step(self, val_loader):
        """Validation loop to calculate the loss on the validation set."""
        self.dino.student.module.eval()
        self.dino.teacher.eval()
        val_loss = 0.0
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation')
        with torch.no_grad():
            for i, input_dict in progress_bar:
                student_outputs, teacher_outputs = [], []
                for j, image in enumerate(input_dict['x'][1]):
                    image = image.to(self.self.device)
                    student_output = self.dino(image)
                    student_outputs.append(student_output)
                    if j < 2:
                        teacher_output = self.dino.forward(image, is_teacher=True)
                        teacher_outputs.append(teacher_output)

                student_outputs_tensor = torch.stack(student_outputs)
                teacher_outputs_tensor = torch.stack(teacher_outputs)
                loss = self.loss(
                    student_output=student_outputs_tensor,
                    teacher_output=teacher_outputs_tensor,
                    epoch=-1)  # Validation

                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        return val_loss

    def training_loop(self, train_loader, knn_val, knn_train, epochs):
        self.log('------------------- Initial KNN Validation --------------------')
        initial_top1_acc, initial_top2_acc = self.knn_validation(knn_train, knn_val, k_values=range(1, 20), T=0.07)
        self.best_top1_acc = initial_top1_acc[1]
        self.log(f'Initial Top1 accuracy: {self.best_top1_acc:.4f}')

        """Complete training loop to train and validate the model."""
        for epoch in range(epochs):
            self.log(f'------------------- Epoch:{epoch} --------------------')
            start_time = time.time()
            # Perform a training step
            train_loss = self.training_step(train_loader, epoch)
            top1_acc, top2_acc = self.knn_validation(knn_train, knn_val, k_values=range(1, 20), T=0.07)

            # Log the training loss
            self.log(f'\nTraining loss: {train_loss:.4f}')
            end_time = time.time()
            self.log(f'Time elapse: {end_time - start_time:.2f} seconds')

            # Log the current time
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            self.log(f'Current time: {current_time}')

            # Check if the current top1 accuracy is the best
            if top1_acc[1] > self.best_top1_acc:
                if top1_acc[1] > self.best_top1_acc:
                    self.best_top1_acc = top1_acc[1]

                self.log(f'***New best Top1 (KNN) accuracy at epoch {epoch}! Saving checkpoint***')
                model_suffix = f'DINO_BASE_Ep_{str(epoch).zfill(3)}_Acc{top1_acc[1]:.2f}.pth'
                # Save the model: teacher & student
                torch.save(
                    self.dino.student.module.state_dict(), f'{self.Log.trained_dir}/Student_{model_suffix}')
                torch.save(
                    self.dino.teacher.state_dict(), f'{self.Log.trained_dir}/Teacher_{model_suffix}')


if __name__ == '__main__':
    # Get the root directory of the experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    add_dict_to_argparser(parser, default_dict={**load_yaml(f'{exp_root}/configs/dinotrain/{args.config}.yaml')})
    args = parser.parse_args()


    ################################ Load the dataset ###################################
    # TODO: Replace DATA_PATH with the path to the dataset.
    # TODO:
    #   DATA_PATH[0]: Path to unlabeled frames. DINO training only requires these frames.
    #                 Labeled frames are randomly sampled from the ultrasound video.
    #   DATA_PATH[1]: Path to labeled frames. These are used for KNN validation.
    # TODO: Download the dataset from the link provided in the README.md.
    #####################################################################################
    DATA_PATH = (
        f"{root}/data/15_frames_processed/",
        f"{root}/data/labeled_frames/",
    )
    #####################################################################################

    out_name = (f"Medino-{args.note}-Arch{args.model_type}-It{args.iters}-B{args.batch_size}-Mo{args.init_momentum}"
                f"-CUDA{args.CUDA}-{time.strftime('%m%d_%H%M_%S', time.localtime())}")
    Medino = Medino_train(
        args=parser.parse_args(),
        folder_name=out_name,
        device=device
    )

    aug_dict = Medino.make_augmentation(args.local_crops_number)
    dataloader_dict = Medino.make_dataset(DATA_PATH, aug_dict, fold=args.fold)

    Medino.training_loop(
        **dataloader_dict,
        epochs=args.epochs,
    )
