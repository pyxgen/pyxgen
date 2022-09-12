from clip.model import CLIP as ClipModel
import torch
from torch.nn import CosineSimilarity, Sigmoid
from torchvision.transforms import Compose, ToPILImage, RandomCrop, Resize, ColorJitter, RandomPerspective

torch.manual_seed(17)


def generator_with_transforms(model: ClipModel, preprocess: Compose, unbounded_image: torch.Tensor, text_features: torch.Tensor):

    optimizer = torch.optim.Adam([unbounded_image], lr=0.03)
    n_iterations = 10000
    cos = CosineSimilarity()
    normalize = preprocess.transforms[-1]
    to_pil = ToPILImage()

    transformer = RandomTransforms(n_transforms=10, crop_size=(model.visual.input_resolution, model.visual.input_resolution))

    for i in range(n_iterations):

        image = Sigmoid()(unbounded_image)  # Bound the image
        loss = 0

        repeated_text_features = text_features.repeat(transformer.n_transforms, 1)
        transformed_images = transformer.apply_random_transforms(image)

        normalized_crops = normalize(transformed_images)

        crops_features = model.encode_image(normalized_crops)
        loss = -cos(repeated_text_features, crops_features).sum() / transformer.n_transforms

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with torch.no_grad():

                resized_image = Resize(model.visual.input_resolution)(image)
                normalized_image = normalize(resized_image)
                image_features = model.encode_image(normalized_image)
                similarity = cos(text_features, image_features)
                print(f"Similarity between text encoding and image encoding: {similarity.item():.4f}")
                print(image.mean(axis=(2, 3)))

                pil_image = to_pil(image.squeeze(0))
                pil_image.show()


class RandomTransforms:
    def __init__(self, n_transforms, crop_size) -> None:

        self.n_transforms = n_transforms
        self.crop_size = crop_size

    def apply_random_transforms(self, image: torch.Tensor):
        repeated_images = image.repeat(self.n_transforms, 1, 1, 1)
        transformed_images = RandomCrop(self.crop_size)(repeated_images)
        transformed_images = ColorJitter()(transformed_images)
        transformed_images = RandomPerspective()(transformed_images)

        return transformed_images
