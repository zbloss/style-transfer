import torch
import pytorch_lightning as pl
from .image_dataset import ImageDataset


class DataModule(pl.LightningDataModule):

    def __init__(self, user_image: torch.Tensor, style_image: torch.Tensor):
        super().__init__()
        self.user_image = user_image
        self.style_image = style_image

    def train_dataloader(self):
        self.dataset = torch.utils.data.TensorDataset(self.user_image, self.style_image)
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=1)
    