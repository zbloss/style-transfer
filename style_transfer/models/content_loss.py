import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """Helper class that wraps MSE Loss into
    a PyTorch Module.
    """

    def __init__(self, target: torch.Tensor):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, logits):
        """
        Calculates MSE Loss given the input `logits`
        and the provided true label, `self.target`.

        Args:
            logits (torch.Tensor): Model output prediction.

        Returns:
            logits (torch.Tensor): Model output prediction.
        """

        self.loss = F.mse_loss(logits, self.target)
        return logits
