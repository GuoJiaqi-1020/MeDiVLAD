import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Medino.dino_utils.vision_transformer import DINOHead, vit_small, vit_tiny, vit_base, vit_tinyer


def find_pth_file(exp_root, classifier_name):
    # Check the primary path first
    primary_path = f'{exp_root}/dino_checkpoint/classifier/{classifier_name}.pth'
    if os.path.exists(primary_path):
        return primary_path

    # Check the primary path first
    primary_path2 = f'{exp_root}/dino_checkpoint/backbone/{classifier_name}.pth'
    if os.path.exists(primary_path2):
        return primary_path2

    # If not found, search in all subfolders under trained_model
    trained_model_root = f'{exp_root}/trained_model/'
    for root, dirs, files in os.walk(trained_model_root):
        for file in files:
            if file == f'{classifier_name}.pth':
                return os.path.join(root, file)

    return None  # Return None if file is not found


class DINOLoss(nn.Module):
    """Class to define DINO loss."""

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center.to(teacher_output.device)) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update the center used for teacher outputs."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = (self.center.to(teacher_output.device) * self.center_momentum +
                       batch_center * (1 - self.center_momentum))


class DINO(nn.Module):
    """DINO model class."""

    def __init__(self, out_dim=65536, use_bn=False, model_type="tiny", **kwargs):
        super().__init__()
        model_map = {
            'tiny': vit_tiny(**kwargs),
            'small': vit_small(**kwargs),
            'base': vit_base(**kwargs),
            'vit_tinyer': vit_tinyer(**kwargs)}

        student = model_map[model_type]
        embed_dim = student.embed_dim

        self.student = nn.Sequential(
            student,
            DINOHead(embed_dim, out_dim, use_bn)
        )
        self.student = nn.DataParallel(self.student)

        teacher = model_map[model_type]
        self.teacher = nn.Sequential(
            copy.deepcopy(teacher),
            DINOHead(embed_dim, out_dim, use_bn)
        )

        self.teacher = nn.DataParallel(self.teacher).module
        self._initialize_teacher()

    def _initialize_teacher(self):
        self.teacher.load_state_dict(self.student.module.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, is_teacher=False):
        """Forward pass through the model."""
        batch_size = x.shape[0]
        if not is_teacher:
            x = self.student.module[0](x)
            x = x.view(batch_size, 1, -1)
            x = self.student.module[1](x)
        else:
            with torch.no_grad():
                x = self.teacher[0](x)
                x = x.view(batch_size, 1, -1)
                x = self.teacher[1](x)
        return x

    def get_last_selfattention(self, x):
        """Get the last self-attention layer from the student model."""
        try:
            return self.teacher[0].get_last_selfattention(x)
        except:
            raise Exception


class DownStream(nn.Module):
    def __init__(self,
                 backbone,
                 classifier,
                 backbone_type='dino',
                 checkpoint: str = None,
                 train_classifier: bool = False,
                 train_backbone: bool = False,
                 device='cuda',
                 **kwargs):
        super().__init__()
        self.backbone_flag = True
        self.classifier_flag = True

        self.exp_root = self.find_project_root()
        self.logger = kwargs.get('logger', None)

        checkpoint_path = find_pth_file(self.exp_root, checkpoint)
        self.classifier = self.load_pretrained_classifier(checkpoint_path, classifier).to(device)

        if backbone_type == 'dino':
            self.backbone = self.load_pretrained_dinobackbone(checkpoint_path, backbone).to(device)
        else:
            raise ValueError('Invalid backbone type')

        self.backbone = nn.DataParallel(self.backbone)
        self.classifier = nn.DataParallel(self.classifier)

        self.log(
            f'\n#### SUMMARY ####\n'
            f'BACKBONE: {backbone_type} -> Pretrained: {self.backbone_flag} | Checkpoint {checkpoint}\n'
            f'CLASSIFIER -> Pretrained: {self.classifier_flag} | Checkpoint {checkpoint}'
        )

        self.train_classifier = train_classifier
        self.train_backbone = train_backbone
        self.log(f'Train Classifier: {self.train_classifier} | Train Backbone: {self.train_backbone}')

        if train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def find_project_root(target_folder='MeDiVLAD'):
        current_path = os.path.abspath(__file__)
        while True:
            parent_path, current_folder = os.path.split(current_path)
            if current_folder == target_folder:
                return current_path
            if parent_path == current_path:  # Reached the root directory
                raise FileNotFoundError(f"Target folder '{target_folder}' not found")
            current_path = parent_path

    def log(self, message: str):
        if self.logger is not None:
            self.logger.log(message)
        else:
            print(message)

    def load_pretrained_dinobackbone(self, checkpoint: str, dino_backbone):
        self.log(f'\n#### Loading pre-trained BACKBONE ####')
        try:
            state_dict = torch.load(checkpoint)
            if 'backbone' in state_dict:
                state_dict = state_dict['backbone']
            try:
                self.log(f'# DINO BACKBONE -> Try to Load: {checkpoint}')
                dino_backbone.load_state_dict(state_dict)
            except Exception as e:
                self.log(f'# DINO BACKBONE -> Try to Load: {checkpoint} AGAIN')
                dino_backbone[0].load_state_dict(state_dict)
        except Exception as final_e:
            self.backbone_flag = False
            self.log(f'# Error loading {checkpoint} *** ')
            self.log('# Randomly initializing DINO backbone ***')
        return dino_backbone[0]

    def load_pretrained_classifier(self, checkpoint: str, linear_classifier):
        self.log(f'\n#### Loading pre-trained CLASSIFIER ####')
        try:
            self.log(f'# DINO CLASSIFIER -> Try to Load: {checkpoint}')
            state_dict = torch.load(checkpoint)
            linear_classifier.load_state_dict(state_dict['linear_state_dict'])
        except Exception as final_e:
            self.classifier_flag = False
            self.log(f'# Error loading {checkpoint} *** ')
            self.log('# Randomly initializing DINO model ***')
        return linear_classifier

    def extract_img_representation(self, x):
        if self.train_backbone:
            self.backbone.train()
            x = self.backbone(x)
        else:
            with torch.no_grad():
                self.backbone.eval()
                x = self.backbone(x)
        if len(x.shape) >= 2:
            x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        if self.train_backbone:
            # self.backbone.train()
            x = self.backbone(x)
        else:
            with torch.no_grad():
                # self.backbone.eval()
                x = self.backbone(x)
        if len(x.shape) >= 2:
            x = x.squeeze(-1).squeeze(-1)

        if self.train_classifier:
            # self.classifier.train()
            x = self.classifier(x)
        else:
            with torch.no_grad():
                # self.classifier.eval()
                x = self.classifier(x)

        return x


if __name__ == '__main__':
    kwargs = {}
    model = vit_small(**kwargs)
