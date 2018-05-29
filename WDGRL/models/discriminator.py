"""Discriminator for ARDA.

estimate the Wasserstein distance between the source and target
representation distributions
"""


from torch import nn


class Discriminator(nn.Module):
    """Discriminator model."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(980, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out.view(-1)
