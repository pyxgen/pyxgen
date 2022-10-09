import kornia
from kornia.augmentation import RandomPerspective, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomAffine, CenterCrop, RandomResizedCrop
import torch
from torch import nn
from typing import Dict, Optional


class RandomTransforms:
    def __init__(self, n_crops, crop_size, intensity=1, zoom=True) -> None:

        self.n_crops = n_crops
        self.cutn_zoom = int(0.6 * n_crops)
        self.crop_size = crop_size
        self.intensity = 1
        self.zoom = zoom
        self.noise_fac = 0.1

    def apply_random_transforms(self, image: torch.Tensor):

        pooled_image = (nn.AdaptiveAvgPool2d(self.crop_size)(image) + nn.AdaptiveMaxPool2d(self.crop_size)(image)) / 2
        repeated_images = pooled_image.repeat(self.n_crops, 1, 1, 1)

        batch1 = repeated_images[: self.cutn_zoom]
        batch2 = repeated_images[self.cutn_zoom :]

        batch1 = self._zoom_transforms(batch1)
        batch2 = self._wide_transforms(batch2)

        transformed_images = torch.cat([batch1, batch2])

        facs = transformed_images.new_empty([self.n_crops, 1, 1, 1]).uniform_(0, self.noise_fac)
        transformed_images = transformed_images + facs * torch.randn_like(transformed_images)
        # transformed_images = GaussianBlur(3, sigma=(0.01, 0.05))(transformed_images)

        return transformed_images

    def _zoom_transforms(self, transformed_images):
        transformed_images = RandomPerspectivePadded(distortion_scale=0.4, p=0.7)(transformed_images)
        transformed_images = RandomResizedCrop(
            size=self.crop_size,
            scale=(0.25, 0.95),
            ratio=(0.85, 1.2),
            cropping_mode="resample",
            p=1.0,
        )(transformed_images)
        transformed_images = ColorJitter(hue=0.1, saturation=0.1, p=0.8)(transformed_images)
        return transformed_images

    def _wide_transforms(self, transformed_images):
        affine_scale = 0.95
        affine_translate = (1 - affine_scale) / 2
        transformed_images = RandomAffine(degrees=0, translate=(affine_translate, affine_translate), scale=(affine_scale, affine_scale), p=1.0)(
            transformed_images
        )
        transformed_images = CenterCrop(size=self.crop_size, cropping_mode="resample", p=1.0)(transformed_images)
        transformed_images = RandomPerspectiveFilled(distortion_scale=0.2, p=0.7)(transformed_images)
        transformed_images = ColorJitter(hue=0.1, saturation=0.1, p=0.8)(transformed_images)
        return transformed_images


class RandomPerspectivePadded(RandomPerspective):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags,
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        random_padding_mode = "reflection" if bool(torch.bernoulli(torch.Tensor([0.5]))) else "border"
        return kornia.geometry.transform.warp_perspective(
            input,
            transform,
            (height, width),
            mode="bilinear",
            align_corners=False,
            padding_mode=random_padding_mode,
        )


class RandomPerspectiveFilled(RandomPerspective):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags,
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        random_fill_color = torch.rand(3)
        return kornia.geometry.transform.warp_perspective(
            input,
            transform,
            (height, width),
            mode="bilinear",
            align_corners=False,
            padding_mode="fill",
            fill_value=random_fill_color,
        )
