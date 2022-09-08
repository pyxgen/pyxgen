from typing import Tuple

import clip
import torch
from clip.model import CLIP as ClipModel
from torchvision.transforms.transforms import Compose


def load_clip_model(model_name="ViT-B/32",
                    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu") -> \
        Tuple[ClipModel, Compose]:
    """Loads a CLIP model, together with its associated preprocessing steps.

    :param model_name: The name of the CLIP model to load. See clip.available_models() for a complete list of available
                       models.
    :param device: The device (cpu, cuda, ...) on which the operations take place.
    :return: A tuple containing the loaded CLIP model and the associated Compose transform to apply to the images input
             to the image encoder.
    """

    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def encode_text(text: str, model: ClipModel, device: torch.device | str) -> torch.Tensor:
    """Encodes a given text using a given CLIP model.

    :param text: The text to encode.
    :param model: The model which encodes the text.
    :param device: The device (cpu, cuda, ...) on which the operations take place.
    :return: The features corresponding to the encoded text.
    """

    tokenized_text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)

    return text_features


def encode_image(image: torch.Tensor, model: ClipModel) -> torch.Tensor:
    """Encodes a given text using a given CLIP model.

    :param image: The pre-processed image to encode.
    :param model: The model which encodes the image.
    :return: The features corresponding to the encoded image.
    """

    image_features = model.encode_image(image)

    return image_features
