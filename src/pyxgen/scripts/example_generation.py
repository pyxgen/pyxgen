"""
Example image generation using CLIP.

Usage:
    example_test_encoding.py <text> [--clip-model=<name>] [--init=<name>] [--generator=<name>]

Options:
    -h --help             Show this screen.
    --clip-model=<name>   Specify the CLIP model to use to encode the text and the generated image. The possible model names
                          are 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14' and
                          'ViT-L/14@336px'. All CLIP models have an image and a text encoder, and are specified by their
                          image encoder's name.
    --init=<name>         Specify how the image is initialized. Possible init values are 'normal' and 'zeros'.
    --generator=<name>    Specify which generator to use. Possible generators are 'baseline' and 'vqgan'.
"""

import clip
import torch
from docopt import docopt

from pyxgen.clip_utils import load_clip_model, encode_text
from pyxgen.image_generation import baseline_generator, vqgan_generator
from PIL import Image
from torchvision.transforms import Compose


def main():
    arguments = docopt(__doc__)

    text = arguments["<text>"]

    model_name = arguments["--clip-model"]
    if model_name is None:
        model_name = "ViT-B/32"

    if model_name not in clip.available_models():
        raise ValueError(f"No CLIP model named {model_name}")

    print(f'Using model {model_name} to encode text "{text}"')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(model_name, device=device)
    text_features = encode_text(text, model, device)

    print(f"Text encoding shape: {text_features.shape}")

    print("Initializing image")

    # resolution = model.visual.input_resolution
    resolution = 333
    print(resolution)

    init = arguments["--init"]
    if init is None:
        init = "normal"

    if init == "normal":
        dummy_image = torch.normal(mean=0.0, std=1.0, size=(1, 3, resolution, resolution), device=device, requires_grad=True)
    elif init == "zeros":
        dummy_image = torch.zeros(size=(1, 3, resolution, resolution), device=device, requires_grad=True)
    elif init == "perlin":
        preprocess_without_resize = Compose(preprocess.transforms[2:])
        dummy_image = preprocess_without_resize(Image.open("sample_images/perlin333.png")).unsqueeze(0).to(device)
    else:
        raise ValueError(f"No image initialization scheme named {init}")

    print(dummy_image.shape)

    generator_name = arguments["--generator"]
    if generator_name is None:
        generator_name = "baseline"

    if generator_name == "baseline":
        baseline_generator(model, preprocess, dummy_image, text_features)
    elif generator_name == "vqgan":
        vqgan_generator(model, preprocess, dummy_image, text_features, device=device)


if __name__ == "__main__":
    main()
