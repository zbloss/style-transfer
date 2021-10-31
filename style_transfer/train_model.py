import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from data.image_processor import ImageProcessor
from data.data_module import DataModule
from models.style_transfer_model import StyleTransferModel
from PIL import Image
from models.style_loss import StyleLoss
from models.content_loss import ContentLoss
from models.normalization import Normalization
from tqdm import tqdm

# image_processor = ImageProcessor(maximum_image_size=(512,512))
# base_path = 'C:\\Users\\altoz\\Documents\\Projects\\style-transfer\\images'
# style_image = image_processor.load_image(os.path.join(base_path, 'StarryNight.jpg'))
# content_image = image_processor.load_image(os.path.join(base_path, 'City.jpg'))
# image_size = image_processor.get_common_image_size(style_image, content_image)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# user_image = image_processor.prepare_images(content_image, image_size)
# input_image = image_processor.prepare_images(content_image, image_size)
# style_image = image_processor.prepare_images(style_image, image_size)
# normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(image_processor.device)
# normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(image_processor.device)

# datamodule = DataModule(user_image, style_image)
# model = StyleTransferModel(
#     user_image=user_image,
#     style_image=style_image,
#     normalization_mean=normalization_mean,
#     normalization_std=normalization_std
# )

# cnn = models.vgg19(pretrained=True).features.to(device).eval()

# model._build_model_and_loss_functions(
#     base_cnn_model=cnn
# )

# print('Training...')
# tb_logger = pl_loggers.TensorBoardLogger('logs/')
# trainer = pl.Trainer(
#     max_epochs=4500,
#     gpus=1,
#     logger=tb_logger
# )
# trainer.fit(model, datamodule)

# sample = user_image.to(device)
# model.to(device)

# image = model(sample)
# image_processor.save_image(image, os.path.join(base_path, 'output.png'), 'Sample Output', display_image=True)


def train_model(user_image: Image, style_image: Image):
    """Trains a Deep Learning model to extract the stylings
    from `style_image` and applies them onto `user_image`
    then returns `user_image`.

    Args:
        user_image (Image): Image you want to apply styles onto.
        style_image (Image): Image you want to extract styles from.

    Returns:
        user_image (Image): `user_image` with styling applied.

    """

    image_processor = ImageProcessor(maximum_image_size=(512, 512))
    print(f"user_image.size: {user_image.size} | style_image.size: {style_image.size}")

    image_size = image_processor.get_common_image_size(user_image, style_image)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    user_image = image_processor.prepare_images(user_image, image_size)
    style_image = image_processor.prepare_images(style_image, image_size)
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(image_processor.device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(image_processor.device)

    datamodule = DataModule(user_image, style_image)
    model = StyleTransferModel(
        user_image=user_image,
        style_image=style_image,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    model._build_model_and_loss_functions(base_cnn_model=cnn)

    print("Training...")
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    trainer_params = {"max_epochs": 6000}

    if device != "cpu":
        print("Using a gpu!")
        trainer_params["gpus"] = 1

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, datamodule)

    sample = user_image.to(device)
    model.to(device)

    image = model(sample)
    image = image_processor.save_image(
        image, "output.png", "Sample Output", display_image=True
    )
    return image
