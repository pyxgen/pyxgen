from clip.model import CLIP as ClipModel
import torch
from torch.nn import CosineSimilarity, Sigmoid
from torchvision.transforms import Compose, ToPILImage


def baseline_generator(model: ClipModel,
                       preprocess: Compose,
                       unbounded_image: torch.Tensor,
                       text_features: torch.Tensor):

    optimizer = torch.optim.Adam([unbounded_image], lr=0.01)
    n_iterations = 10000
    cos = CosineSimilarity()
    normalize = preprocess.transforms[-1]
    to_pil = ToPILImage()

    for i in range(n_iterations):
        image = Sigmoid()(unbounded_image)  # Bound the image
        normalized_image = normalize(image)
        image_features = model.encode_image(normalized_image)
        loss = - cos(text_features, image_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            with torch.no_grad():
                image_features = model.encode_image(normalized_image)
                similarity = cos(text_features, image_features)
                print(f"Similarity between text encoding and image encoding: {similarity.item():.4f}")
                print(image.mean(axis=(2, 3)))

                pil_image = to_pil(image.squeeze(0))
                pil_image.show()

            

