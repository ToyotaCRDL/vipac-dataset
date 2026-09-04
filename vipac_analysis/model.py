"""RSS-CNN model for vehicle impression ranking."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def softrelu(x, beta=1.0):
    return F.softplus(beta * x, beta=beta) / beta


def ranking_loss(outputs_lr, outputs_rr, labels, lmd, device):
    """Margin-based ranking loss (squared hinge).

    min2019_multi Eq.3: sum(max(0, 1 + rr - lr)^2) * lmd * direction
    """
    lr = outputs_lr.squeeze(-1)  # (B, 1) -> (B,)
    rr = outputs_rr.squeeze(-1)  # (B, 1) -> (B,)
    labels = labels.float()
    direction = (0.5 - labels) * 2
    margin = 1 + rr - lr
    x = torch.max(torch.tensor([0], dtype=torch.float32, device=device),
                  margin * direction)
    return torch.sum(x * x) * lmd


class RSSCNN(nn.Module):
    """Ranking Streetscore-CNN with ResNet backbone.

    Args:
        cnn_name: 'resnet18', 'resnet34', or 'resnet50'
        fdb_path: Path to FractalDB pre-trained weights (None = ImageNet)
        gap: Use global average pooling for ranking branch
        activation: 'softrelu' or 'sigmoid'
        catdim: Concat dimension for pair features (1=batch, 2=channel)
    """

    def __init__(self, cnn_name='resnet34', fdb_path=None,
                 gap=True, activation='sigmoid', catdim=2):
        super().__init__()
        self.catdim = catdim

        cnndim_map = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}
        self.cnndim = cnndim_map[cnn_name]
        self.fcdim = 512 * 1 * 1 if catdim == 1 else 512 * 8 * 1

        # Backbone
        cnn_fn = getattr(models, cnn_name)
        if fdb_path:
            import os
            if os.path.exists(fdb_path):
                self.cnn = cnn_fn(weights=None)
                self.cnn.load_state_dict(torch.load(fdb_path, map_location='cpu', weights_only=True))
                print(f'Loaded FractalDB weights from {fdb_path}')
            else:
                print(f'Warning: FractalDB path not found: {fdb_path}. '
                      f'Using ImageNet weights.')
                weights_fn = getattr(models, f'ResNet{cnn_name[6:].upper()}_Weights')
                try:
                    self.cnn = cnn_fn(weights=getattr(weights_fn, 'IMAGENET1K_V1'))
                except AttributeError:
                    self.cnn = cnn_fn(weights=None)
        else:
            weights_fn = getattr(models, f'ResNet{cnn_name[6:].upper()}_Weights')
            try:
                self.cnn = cnn_fn(weights=getattr(weights_fn, 'IMAGENET1K_V1'))
            except AttributeError:
                self.cnn = cnn_fn(weights=None)

        # Features: everything except last avgpool and fc
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-2])

        # Classification head (left vs right comparison)
        if catdim == 1:
            self.conv1 = nn.Conv2d(self.cnndim * 2, 512, kernel_size=3, padding=0)
        else:
            self.conv1 = nn.Conv2d(self.cnndim, 512, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.fc = nn.Linear(self.fcdim, 2)

        # Ranking head (single image score)
        self._gap = gap
        self._activation = activation

        if gap:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.rank_fc = nn.Linear(self.cnndim, 1)
        else:
            self.rank_fc1 = nn.Linear(self.cnndim * 7 * 7, 4096)
            self.rank_fc2 = nn.Linear(4096, 4096)
            self.rank_fc3 = nn.Linear(4096, 1)

    def _rank(self, feature):
        """Compute single-image impression score from feature map."""
        if self._gap:
            x = self.avgpool(feature)
            x = torch.flatten(x, 1)
            if self._activation == 'softrelu':
                return softrelu(self.rank_fc(x))
            else:
                return torch.sigmoid(self.rank_fc(x))
        else:
            x = softrelu(self.rank_fc1(feature.view(-1, self.cnndim * 7 * 7)))
            x = softrelu(self.rank_fc2(x))
            if self._activation == 'softrelu':
                return softrelu(self.rank_fc3(x))
            else:
                return torch.sigmoid(self.rank_fc3(x))

    def forward(self, left_img, right_img=None):
        """Forward pass.

        If right_img is None: return left image score (scoring mode).
        If right_img is given: return (comparison, left_score, right_score).
        """
        left_feature = self.cnn_features(left_img)

        if right_img is None:
            return self._rank(left_feature)

        right_feature = self.cnn_features(right_img)

        # Classification: concatenate features and predict which is preferred
        x = torch.cat((left_feature, right_feature), dim=self.catdim)
        x = softrelu(self.conv1(x))
        x = softrelu(self.conv2(x))
        x = softrelu(self.conv3(x))
        x = x.view(-1, self.fcdim)
        comparison = self.fc(x)

        lr = self._rank(left_feature)
        rr = self._rank(right_feature)

        return comparison, lr, rr
