"""Minimal image-prep helpers vendored from pix2pix-turbo."""


def build_transform(image_prep):
    """Construct an image transform pipeline for inference."""
    from PIL import Image
    from torchvision import transforms

    if image_prep == "resized_crop_512":
        return transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(512),
            ]
        )
    if image_prep == "resize_286_randomcrop_256x256_hflip":
        return transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if image_prep in {"resize_256", "resize_256x256"}:
        return transforms.Compose(
            [transforms.Resize((256, 256), interpolation=Image.LANCZOS)]
        )
    if image_prep in {"resize_512", "resize_512x512"}:
        return transforms.Compose(
            [transforms.Resize((512, 512), interpolation=Image.LANCZOS)]
        )
    if image_prep == "no_resize":
        return transforms.Lambda(lambda x: x)
    raise ValueError(f"Unsupported image_prep value: {image_prep}")
