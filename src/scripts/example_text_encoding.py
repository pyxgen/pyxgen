"""
Example text encoding using CLIP.

Usage:
    example_test_encoding.py <text>
    example_test_encoding.py <text> --model=<name>

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

from clip_utils import encode_text, load_clip_model
from docopt import docopt


def main():
    arguments = docopt(__doc__)
    text = arguments['<text>']
    model_name = arguments['--model']
    if model_name is None:
        model_name = "ViT-B/32"

    if model_name not in clip.available_models():
        raise ValueError('No CLIP model named {}'.format(model_name))

    print('Using model {} to encode text "{}"'.format(model_name, text))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = load_clip_model(model_name, device=device)
    text_features = encode_text(text, model, device)

    print("Text encoding shape: {}".format(text_features.shape))


if __name__ == "__main__":
    main()
