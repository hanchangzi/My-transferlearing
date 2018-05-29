"""Generator for ARDA.

learn the domain-invariant feature representations from inputs across domains.
"""

from torch import nn


class Generator(nn.Module):
    """LeNet encoder model for ARDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Generator, self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            # 输入和输出通道数分别为10和20  20*14*14
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),  # 输入和输出通道数分别为10和20  20*7*7
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = conv_out.view(conv_out.size(0),-1)
        return feat
