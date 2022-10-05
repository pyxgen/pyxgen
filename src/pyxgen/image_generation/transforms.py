from kornia.augmentation import RandomPerspective, RandomCrop, RandomHorizontalFlip
import torch


class RandomTransforms:
    def __init__(self, n_crops, crop_size) -> None:

        self.n_crops = n_crops
        self.crop_size = crop_size

    def apply_random_transforms(self, image: torch.Tensor):
        repeated_images = image.repeat(self.n_crops, 1, 1, 1)
        transformed_images = RandomCrop(self.crop_size)(repeated_images)
        # transformed_images = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.1)(transformed_images)
        transformed_images = RandomPerspective(distortion_scale=0.8, p=1.0)(transformed_images)
        transformed_images = RandomHorizontalFlip(p=0.5)(transformed_images)
        # transformed_images = GaussianBlur(3, sigma=(0.01, 0.05))(transformed_images)

        return transformed_images
