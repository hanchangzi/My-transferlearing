"""Classifier for ARDA.

guarantee learned domain-invariant representations are discriminative enough
to accomplish the final classification task
"""

import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """LeNet classifier model for ARDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Classifier, self).__init__()
        self.fc2=nn.Sequential(
            nn.Linear(980, 400),  # 输入的向量大小和输出的大小分别为320和50
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(400, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10)
        )
    def forward(self, feat):
        # print(feat.shape)
        """Forward the LeNet classifier."""
        out = self.fc2(feat)
        return out
