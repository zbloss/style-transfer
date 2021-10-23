import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import pytorch_lightning as pl
from .style_loss import StyleLoss
from .content_loss import ContentLoss
from .normalization import Normalization


class StyleTransferModel(pl.LightningModule):
    def __init__(
        self,
        user_image: torch.Tensor,
        style_image: torch.Tensor,
        normalization_mean: torch.tensor,
        normalization_std: torch.tensor,
        content_layers: list = ['conv_4'],
        style_layers: list = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
        learning_rate: float = 0.003,
        style_weight: int = 1000000,
        content_weight: int = 1,
        device: str = None
    ):
        """
        
        Args:
            user_image (torch.Tensor): The image you want to apply stylings to.
            style_image (torch.Tensor): The iamge you want to extract styleings 
                                            from.
            normalization_mean (torch.Tensor): Mean Values to normalize on.
            normalization_std (torch.Tensor): Standard Deviation Values to 
                                                normalize on.
            content_layers (list): Names of layers you want to extract content
                                        from.
            style_layers (list): Names of layers you want to extract stylings
                                        from.
            style_weight (int): Amount of weight to apply to styling loss.
            content_weight (int): Amount of weight to apply to content loss.
            learning_rate (float): Learning Rate to use during training.
            device (str): Optional string representing what device to use. If
                            None, it will attempt to load to cuda if available.

        Returns:
            None
        
        """
        super().__init__()

        self.user_image = user_image
        self.user_image.requires_grad_(True)
        self.style_image = style_image
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.learning_rate = learning_rate
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalization = Normalization(normalization_mean, normalization_std) #.to(self.device)

    def _build_model_and_loss_functions(self, base_cnn_model: nn.Sequential):
        """
        Helper method to construct our model from a pretrained
        model. Also builds the necessary content and style loss
        functions.

        Args:
            base_cnn_model (nn.Sequential): A CNN that you want to extract layers 
                                                from to build your model from.
        Returns:
            model (nn.Sequential): Model object ready for training. Stored as [`self.model`]
            content_losses (List[torch.Tensor]): List of losses to track in content layers.
                                                    Stored as [`self.content_losses`]
            style_losses (List[torch.Tensor]): List of losses to track in style layers.
                                                    Stored as [`self.style_losses`]
        """

        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalization)

        i = 0 
        for layer in base_cnn_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)

            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            # Here we track specific losses for those selected layers.
            if name in self.content_layers:

                target = model(self.user_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:

                target_feature = model(self.style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # removes the remaining layers.
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        model.requires_grad_(False)

        self.model = model
        self.content_losses = content_losses
        self.style_losses = style_losses

        return (self.model, self.content_layers, self.style_losses)

            
    def forward(self, x):
        original_input = x
        x = self.model(x)
        return original_input

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        style_score = 0
        content_score = 0

        for idx, sl in enumerate(self.style_losses):
            style_score += sl.loss

        for idx, cl in enumerate(self.content_losses):
            content_score += cl.loss

        style_score *= self.style_weight
        content_score *= self.content_weight

        loss = style_score + content_score

        self.log('style_loss', style_score)
        self.log('content_loss', content_score)
        self.log('loss', loss)

        if self.global_step % 100 == 0:
            self.logger.experiment.add_image(
                'logged_image',
                y_hat.squeeze(0),
                self.global_step
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.user_image], lr=self.learning_rate)
