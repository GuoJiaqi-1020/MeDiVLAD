import torch
import torch.nn as nn
import torch.nn.functional as F


class DualVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=1.0,
                 normalize_input=True, centroids=None):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(DualVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        if centroids is not None:
            try:
                self.centroids = nn.Parameter(torch.load(f'./centroid/{centroids}'))
            except:
                print('the centroid will be loaded from checkpoint')
                pass

        self.V = torch.nn.Sequential(
            torch.nn.Linear(dim * self.num_clusters, dim * self.num_clusters),
        )
        self.w = torch.nn.Sequential(
            torch.nn.Linear(dim * self.num_clusters, 1),
        )
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def frame_assign(self, x):
        w_Tanh_Vh = self.w(torch.nn.Tanh()(self.V(x))).permute(0, 2, 1)
        frame_assign = F.softmax(w_Tanh_Vh/1.2, dim=-1)
        return frame_assign

    def class_assign(self, x):
        class_assign = self.conv(x).view(x.shape[0], self.num_clusters, -1)
        class_assign = F.softmax(class_assign, dim=1)
        return class_assign

    def forward(self, x):
        b, s, d = x.shape[:3]  # (N, C, H, W)
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)
        x = x.view(b * s, d, 1, 1)  # across descriptor dim (N, C, 1, 1)

        # soft-assignment
        class_assign = self.class_assign(x)

        # calculate residuals to each clusters
        x_flatten = x.view(b * s, d, -1).expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
        centroids = self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual = ((x_flatten - centroids) * class_assign.unsqueeze(2)).sum(dim=-1)

        # soft-assignment frame-level
        residual = residual.view(b, s, -1)  # flatten
        residual = torch.bmm(self.frame_assign(residual), residual).view(b, self.num_clusters, d)

        vlad = F.normalize(residual, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(b, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
