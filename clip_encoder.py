import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def encode_text(text: str, model: clip.model.CLIP):
    tokenized_text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)

    return text_features
