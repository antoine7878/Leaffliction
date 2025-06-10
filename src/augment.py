import random
import skimage


def rotatation(img):
    """Rotation"""
    return skimage.transform.rotate(
        img, random.randrange(1, 35) * 10, preserve_range=True
    ).astype("uint8")


def blur(img):
    """Blur"""
    s = random.uniform(0, 3)
    t = random.uniform(0, 3.5)
    return skimage.filters.gaussian(
        img,
        sigma=s,  # type: ignore
        truncate=t,
        channel_axis=-1,
        preserve_range=True,
    ).astype("uint8")


def contrast(img):
    """Contrast"""
    return (skimage.exposure.equalize_hist(img) * 256).astype("uint8")


def scaling(img):
    """Scalling"""
    return (
        skimage.transform.resize(img[64:192, 64:192], (256, 256)) * 256
    ).astype("uint8")


def illumination(img):
    """Illumination"""
    return skimage.exposure.adjust_gamma(img, gamma=0.5)  # type: ignore


def flip_h(img):
    """Flip_H"""
    return img[:, ::-1]


def flip_v(img):
    """Flip_V"""
    return img[::-1, :]


AUGMENTATIONS = [rotatation, blur, contrast, scaling, illumination, flip_h]
