"""
Example image generation using CLIP.

Usage:
    example_test_encoding.py <text>
    example_test_encoding.py <text> --model=<name>

Options:
    -h --help        Show this screen.
    --model=<name>   Specify the CLIP model to use to encode the text and the generated image. The possible model names
                     are 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14' and
                     'ViT-L/14@336px'. All CLIP models have an image and a text encoder, and are specified by their
                     image encoder's name.
"""

import clip
import torch
from docopt import docopt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from clip_utils import load_clip_model, encode_text
from image_generation import baseline_generator


def main():
    arguments = docopt(__doc__)
    text = arguments['<text>']
    model_name = arguments['--model']
    if model_name is None:
        model_name = "ViT-B/32"

    if model_name not in clip.available_models():
        raise ValueError(f'No CLIP model named {model_name}')

    green_image = Image.open("sample_images/green.png")
    print(green_image.getpixel((100, 100)))
    print(green_image.convert("RGB").getpixel((100, 100)))
    print(ToTensor()(green_image))
    print(ToPILImage()((ToTensor()(green_image))).getpixel((100,100)))
    green_image.show()

    print(f'Using model {model_name} to encode text "{text}"')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(model_name, device=device)
    print(preprocess)
    text_features = encode_text(text, model, device)

    print(f"Text encoding shape: {text_features.shape}")

    print("Initializing image")

    resolution = model.visual.input_resolution

    # dummy_image = torch.normal(mean=0., std=1., size=(1, 3, resolution, resolution),
    #                            device=device, requires_grad=True)
    dummy_image = torch.zeros(size=(1, 3, resolution, resolution), device=device, requires_grad=True)
    baseline_generator(model, preprocess, dummy_image, text_features)


if __name__ == "__main__":
    main()
