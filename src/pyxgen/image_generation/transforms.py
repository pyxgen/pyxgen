from clip.model import CLIP as ClipModel
import torch
from torch.nn import CosineSimilarity, Sigmoid
from torchvision.transforms import Compose, ToPILImage, RandomCrop

torch.manual_seed(17)


def generator_with_transforms(model: ClipModel, preprocess: Compose, unbounded_image: torch.Tensor, text_features: torch.Tensor):

    optimizer = torch.optim.Adam([unbounded_image], lr=0.01)
    n_iterations = 10000
    cos = CosineSimilarity()
    normalize = preprocess.transforms[-1]
    to_pil = ToPILImage()

    n_transforms = 10
    crop_size = (model.visual.input_resolution, model.visual.input_resolution)

    for i in range(n_iterations):

        image = Sigmoid()(unbounded_image)  # Bound the image
        loss = 0

        for transform_i in range(n_transforms):

            random_crop = RandomCrop(crop_size)(image)
            normalized_crop = normalize(random_crop)
            crop_features = model.encode_image(normalized_crop)
            loss += -cos(text_features, crop_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            with torch.no_grad():

                print(image.mean(axis=(2, 3)))

                pil_image = to_pil(image.squeeze(0))
                pil_image.show()
