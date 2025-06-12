from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


def p(df, color):
    df = df[df["color channel"] == color]
    plt.plot(y=df["hist_count"], x=df["pixel intensity"])


def gaussian_blur(img):
    """Gaussian blur"""
    img = img.copy()
    img = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    img = pcv.threshold.binary(gray_img=img, threshold=134, object_type="light")
    img = pcv.gaussian_blur(img=img, ksize=(5, 5))
    return img


def background(img):
    """Background"""
    img = img.copy()
    leaf_mask = get_leaf(img)
    img = pcv.apply_mask(img=img, mask=leaf_mask, mask_color="white")
    return img


def analyse_object(img):
    """Analyse object"""
    img = img.copy()
    leaf_mask = get_leaf(img)
    roi1 = pcv.roi.circle(img=img, x=128, y=128, r=128)
    kept_mask = pcv.roi.filter(mask=leaf_mask, roi=roi1, roi_type="partial")
    img = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    return img


def roi_object(img):
    """Roi object"""
    img = img.copy()
    leaf_mask = get_leaf(img)
    roi1 = pcv.roi.circle(img=img, x=128, y=128, r=128)
    kept_mask = pcv.roi.filter(mask=leaf_mask, roi=roi1, roi_type="cutto")
    img = pcv.analyze.bound_vertical(
        img=img, labeled_mask=kept_mask, line_position=0
    )
    return img


def pseudo_landmark(img):
    """Pseudo landmark"""
    get_leaf(img)
    img = img.copy()
    mask = get_leaf(img)
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=mask
    )
    draw_point(img, top, (255, 0, 0))
    draw_point(img, bottom, (0, 255, 0))
    draw_point(img, center_v, (0, 0, 255))
    return img


def draw_point(img, landmarks, color):
    for points in landmarks:
        for x in range(int(points[0][0] - 4), int(points[0][0] + 5)):
            for y in range(int(points[0][1] - 4), int(points[0][1] + 5)):
                for i, c in enumerate(color):
                    img[x, y, i] = c


def get_leaf(img):
    thresh1 = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )
    img = pcv.fill(bin_img=thresh1, size=50)
    img = pcv.fill_holes(img)
    return img


def histogram(img):
    "Histogram"
    mask = get_leaf(img)
    _, data = pcv.visualize.histogram(img=img, mask=mask, hist_data=True)

    _, ax = plt.subplots(figsize=(10, 6))
    for channel in data["color channel"].unique():
        channel_data = data[data["color channel"] == channel]
        ax.plot(
            channel_data["pixel intensity"],
            channel_data["proportion of pixels (%)"],
            marker="o",
            label=channel,
        )

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion of Pixels (%)")
    ax.set_title("Pixel Intensity Distribution by Color Channel")
    ax.legend(title="Color Channel")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


TRANSFORMS = [
    gaussian_blur,
    background,
    analyse_object,
    roi_object,
    pseudo_landmark,
]
