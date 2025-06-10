from plantcv import plantcv as pcv


def gaussian_blur(img):
    """Gaussian blur"""
    img = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    img = pcv.threshold.binary(
        gray_img=img, threshold=134, object_type="light"
    )
    img = pcv.gaussian_blur(img=img, ksize=(5, 5))
    return img


def mask(img):
    """Mask"""
    return img


def analyse_object(img):
    """Analyse object"""
    thresh1 = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )
    a_fill_image = pcv.fill(bin_img=thresh1, size=50)
    a_fill_image = pcv.fill_holes(a_fill_image)
    roi1 = pcv.roi.circle(img=img, x=128, y=128, r=100)
    kept_mask = pcv.roi.filter(mask=a_fill_image, roi=roi1, roi_type="partial")
    img = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return img


TRANSFORMS = [gaussian_blur, mask, analyse_object]
