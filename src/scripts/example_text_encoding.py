import torch

from clip_utils import encode_text, load_clip_model


def main():
    text = "a cat on a dog on a cat on a dog"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessor = load_clip_model("RN50", device=device)
    text_features = encode_text(text, model, device)
    print(text_features)


if __name__ == "__main__":
    main()
