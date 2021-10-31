import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image


class ImageProcessor:
    def __init__(self, maximum_image_size: Tuple[int, int] = [1080, 1080]):

        if type(maximum_image_size) == list:
            maximum_image_size = tuple(maximum_image_size)
        self.maximum_image_size = maximum_image_size
        self.tensor_to_image = transforms.ToPILImage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_image(self, path_to_image_file: str) -> Image:
        """
        Given a str representing a path to an image file,
        this method loads the file into a Pillow Image and
        returns it.

        Args:
            path_to_image_file (str): Filepath of an image file.

        Returns:
            image (Image): PIL Image file.
        """

        assert os.path.isfile(
            path_to_image_file
        ), f"Provided path_to_image_file does not exists: {path_to_image_file}"
        image = Image.open(path_to_image_file)
        return image

    @staticmethod
    def get_common_image_size(
        first_image: Image, second_image: Image
    ) -> Tuple[int, int]:
        """
        Given two images, determines the smallest dimension
        amongst them.

        Args:
            first_image (Image): A PIL Image.
            second_image (Image): A PIL Image.

        Returns:
            smallest_size (Tuple[int, int]): Tuple containing the
                                                smallest image size x
                                                smallest image size

        """

        first_height, first_width = first_image.size
        second_height, second_width = second_image.size

        smallest_dim = min([first_height, first_width, second_height, second_width])
        return (smallest_dim, smallest_dim)

    def prepare_images(self, image: Image, image_size: Tuple[int, int] = None):
        """
        Given a PIL image this method performs the preprocessing
        needed to use it in the style-transfer model. Scales a
        PIL Image to `image_size` or `self.maximum_image_size`,
        then transforms it into a PyTorch tensor, and lastly
        returns that tensor as a float on `self.device`

        Args:
            image (Image): PIL Image that you want to prepare.

        Returns:
            image (Image): PIL Image ready for modeling.
        """

        if type(image_size) in [int, float]:
            image_size = (int(image_size), int(image_size))

        if image_size is None or image_size > self.maximum_image_size:
            image_size = self.maximum_image_size

        image_scaler = transforms.Compose(
            [transforms.Scale(image_size), transforms.ToTensor()]
        )
        image = image_scaler(image).unsqueeze(0)

        return image.to(self.device, torch.float)

    def save_image(
        self,
        tensor: torch.Tensor,
        filepath: str,
        title: str = None,
        display_image: bool = False,
    ):
        """
        Saves and optionally displays an image
        from a PyTorch Tensor.

        Args:
            tensor (torch.Tensor): Contains the image stored as a PyTorch Tensor.
            filepath (str): Path to where you want to store the image.
            title (str): Optional argument that adds a title to the image.
            display_image (bool): True if you want to display the image.

        Returns:
            image (Image): The PIL Image.
        """

        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.tensor_to_image(image)

        if display_image:
            plt.figure(figsize=(8, 8))
            if title is not None:
                plt.title(title)
            plt.imshow(image)

        try:
            file_directory, _ = os.path.split(filepath)
            if not os.path.exists(file_directory):
                os.makedirs(file_directory)
        except:
            pass
        image.save(filepath)
        return image
