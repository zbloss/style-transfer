import torch
import torch.nn as nn


class Normalization(nn.Module):
    """Normalization layer as a PyTorch Module"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the provided image by subtracting the mean
        and dividing by the standard deviation.

        Args:
            image (torch.Tensor): An image as a PyTorch Tensor.

        Returns:
            image (torch.Tensor): A normalized image as a PyTorch Tensor.
        """
        return (image - self.mean) / self.std
