from clip.model import CLIP as ClipModel
import torch
from torch.nn import CosineSimilarity
from taming.models.vqgan import VQModel
import subprocess
import os
from omegaconf import OmegaConf
from torch.nn import functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, ToPILImage
from pyxgen.image_generation.transforms import RandomTransforms

# Taken from pixray (https://github.com/pixray/pixray)
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


# Taken from pixray (https://github.com/pixray/pixray)
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


# Adapted from pixray (https://github.com/pixray/pixray)
def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}...")
        subprocess.check_output(["wget", "-O", out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


# Adapted from pixray (https://github.com/pixray/pixray)
def quantize(z: torch.Tensor, codebook: torch.Tensor):
    """
    This function quantizes latent representation vectors into their closest (in L2 norm) vectors in a codebook.
    This operation is non-differentiable, so we use replace_grad to make autograd skip from z_q to z when doing the backward pass.
    :param z: The vectors to quantize (shape 1 x h x w x C, with h the height of the latent representation,
     w its width and C the number of channels (or dimensions of the vectors)
    :param codebook: The possible vectors to which we can quantize (shape |Z| x C, where |Z]| is the number of possible vectors in the codebook)
    :return: The quantized vectors (shape 1 x h x w x C)
    """

    # d is the squared L2-norm of the difference between the vectors in z and all vectors in the codebook
    d = (z**2).sum(dim=-1, keepdim=True) + (codebook**2).sum(dim=1) - 2 * z @ codebook.T

    # Select the indices where the L2-distance is minimized
    indices = d.argmin(-1)

    # Quantize z as the codebook entries corresponding to these indices
    z_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook

    # Apply the gradient skipping trick to allow for autograd differentiation
    z_q = replace_grad(z_q, z)
    return z_q


def vqgan_generator(clip_model: ClipModel, preprocess: Compose, initial_image: torch.Tensor, text_features: torch.Tensor, device: torch.device | str):

    vqgan_version = "imagenet_f16_16384"

    # URLs from which we can download the model configurations and parameters
    config_urls = {"imagenet_f16_16384": "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"}
    params_urls = {"imagenet_f16_16384": "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"}
    config_url = config_urls[vqgan_version]
    params_url = params_urls[vqgan_version]

    # Local paths to save the model configurations and parameters
    config_path = f"models/vqgan_{vqgan_version}.yaml"
    params_path = f"models/vqgan_{vqgan_version}.ckpt"

    # Download model configuration and / or parameters, if they are not present locally
    if not os.path.exists("models/"):
        os.makedirs("models/")
    if not os.path.exists(config_path):
        wget_file(config_url, config_path)
    if not os.path.exists(params_path):
        wget_file(params_url, params_path)

    # Load vqgan model
    config = OmegaConf.load(config_path)
    vqgan_model = VQModel(**config.model.params)
    vqgan_model.init_from_ckpt(params_path)
    vqgan_model.to(device).eval()
    vqgan_model.requires_grad_(False)

    image = initial_image
    image.requires_grad_(False)

    n_iterations = 500
    cos = CosineSimilarity()
    to_pil = ToPILImage()

    augmenter = RandomTransforms(n_crops=16, crop_size=(clip_model.visual.input_resolution, clip_model.visual.input_resolution))
    repeated_text_features = text_features.repeat(augmenter.n_crops, 1)

    # Get the initial z-vector from the initial image
    z_vector, emb_loss, info = vqgan_model.encode(image)
    z_vector.requires_grad_(True)

    # We use the same optimization hyper-parameters as in the VQGAN-CLIP paper
    optimizer = torch.optim.Adam([z_vector], lr=0.15, betas=(0.9, 0.999))
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    # z-vector regularization (see section 2.4 from VQGAN-CLIP paper https://arxiv.org/abs/2204.08583)
    # It's not very clearly explained, so we start with a given value for alpha, and decay it such that it reaches a given end value
    # The decay factor is computed dynamically based on values of alpha_begin and alpha_end
    alpha_begin = 0.0
    alpha_end = 0.0
    if alpha_begin == 0.0 or alpha_end == 0.0:
        alpha_decay = 1.0
    else:
        alpha_decay = np.exp((np.log(alpha_end) - np.log(alpha_begin)) / n_iterations)
    alpha = alpha_begin

    resize_to_clip_size = Compose(preprocess.transforms[:2])

    for i in range(n_iterations):
        print(f"Iteration {i}/{n_iterations}")

        z_q = quantize(z_vector.movedim(1, 3), vqgan_model.quantize.embedding.weight).movedim(3, 1)

        image = vqgan_model.decode(z_q)

        # TODO: decide the proper way to normalize the image
        # Normalize the image to the (0, 1) range
        image = (image - image.min()) / (image.max() - image.min())

        # Pixray's way of normalizing (works as well but leads to some saturating pixels)
        # image = (image + 1) / 2.  # If the image is in the range (-1, 1) this places it in the range (0, 1)
        # image = clamp_with_grad(image, 0, 1)  # Clamp the image to the range (0, 1)

        # Using Clip's normalize (does not work)
        # normalize = preprocess.transforms[-1]
        # image = normalize(image)

        transformed_images = augmenter.apply_random_transforms(image)
        crops_features = clip_model.encode_image(transformed_images)

        regularization_loss = alpha * (z_q**2).mean()  # Penalize z_q magnitude in L2 norm
        loss = -cos(repeated_text_features, crops_features).mean() + regularization_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        alpha *= alpha_decay
        lr_scheduler.step()

        if i % 50 == 0:
            with torch.no_grad():
                image_features = clip_model.encode_image(resize_to_clip_size(image))
                similarity = cos(text_features, image_features)
                print(f"Similarity between text encoding and image encoding: {similarity.item():.4f}")
                print(f"Average values of red, green and blue: {image.mean(axis=(2, 3))}")
                for param_group in optimizer.param_groups:
                    print(f"LR = {param_group['lr']}")
                print(f"Alpha (z-vector regularization) = {alpha}")
                print()
                pil_image = to_pil(image.squeeze(0))
                pil_image.show()
