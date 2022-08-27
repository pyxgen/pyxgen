"""
Example text encoding using CLIP.

Usage:
    example_text_encoding.py <text>
    example_text_encoding.py <text> --model=<name>

Options:
    -h --help        Show this screen.
    --model=<name>   Specify the CLIP model to use to encode the text. The possible model names are 'RN50',
                     'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14' and 'ViT-L/14@336px'.
                     All CLIP models have an image and a text encoder, and are specified by their image encoder's name.
                     Here, we are interested in the text encoder, but we still need to specify the corresponding image
                     encoder's name.
"""

import clip
import torch
from docopt import docopt

from clip_utils import encode_text, load_clip_model


def main():
    arguments = docopt(__doc__)
    text = arguments["<text>"]
    model_name = arguments["--model"]
    if model_name is None:
        model_name = "ViT-B/32"

    if model_name not in clip.available_models():
        raise ValueError(f"No CLIP model named {model_name}")

    print(f'Using model {model_name} to encode text "{text}"')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_clip_model(model_name, device=device)
    text_features = encode_text(text, model, device)

    print(f"Text encoding shape: {text_features.shape}")


if __name__ == "__main__":
    main()
