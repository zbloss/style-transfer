import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    """
    Calculates the style transfer loss as a
    PyTorch Module.

    """

    def __init__(self, target: torch.Tensor):
        super(StyleLoss, self).__init__()
        self.target = self.get_gram_matrix(target).detach()

    @staticmethod
    def get_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the Normalized Gram Matrix given a PyTorch Tensor.

        Args:
            tensor (torch.Tensor): Model output prediction.

        Returns:
            normalized_gram_matrix (torch.Tensor): The Gram Matrix computed
                                                    from the model output
                                                    prediction.

        """

        batch_size, feature_maps, feature_map_h, feature_map_w = tensor.size()
        features = tensor.view(batch_size * feature_maps, feature_map_h * feature_map_w)
        gram_matrix = torch.mm(features, features.t())

        normalized_gram_matrix = gram_matrix.div(
            batch_size * feature_maps * feature_map_h * feature_map_w
        )
        return normalized_gram_matrix

    def forward(self, input: torch.Tensor):
        """
        Computes the MSE Loss between the normalized gram matrix
        of the input tensor and the output target.

        Args:
            input (torch.Tensor): The model prediction.

        Returns:
            input (torch.Tensor): The model prediction.
        """

        normalized_gram_matrix = self.get_gram_matrix(input)
        self.loss = F.mse_loss(normalized_gram_matrix, self.target)
        return input
