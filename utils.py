import torch.nn.functional as F


def scale_images(images, image_size=256):
    scaled_images = F.interpolate(
        images,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return scaled_images
