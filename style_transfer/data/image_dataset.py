import torch

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, input_image: torch.Tensor, target_image: torch.Tensor, samples_per_epoch: int = 100):
        self.input_image = input_image.squeeze(0)
        self.target_image = target_image.squeeze(0)
        self.data = [[self.input_image, self.target_image] for _ in range(samples_per_epoch)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
