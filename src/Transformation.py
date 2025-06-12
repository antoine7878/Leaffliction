#!python

import os
import argparse
import warnings
import random
import math
import concurrent.futures

import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from utils import listdir_dirs, listdir_files, n_randchoices
from Distribution import count_class
from transform import TRANSFORMS

warnings.filterwarnings("ignore", message="This might be a color image")


def pars_args():
    parser = argparse.ArgumentParser(
        prog="Transfomation", description="Leaffliction Transfomation program"
    )
    parser.add_argument("file")
    parser.add_argument("-o", "--out", type=str)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    return args


def tranformation():
    args = pars_args()
    if os.path.isfile(args.file):
        assert args.file.lower().endswith(".jpg"), f"{args.file}: wrong format"
        images = transform_file(args.file, args.out, TRANSFORMS, args.save)
        plot_images(images)
    elif os.path.isdir(args.file):
        transform_folder(args)
    else:
        raise TypeError(f"{args.file} not found")


def transform_folder(args):
    counter = count_class(args.file)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    for sub in listdir_dirs(args.file):
        pool.submit(transform_subfolder, counter, sub, args)


def transform_subfolder(counter, sub, args):
    trans = TRANSFORMS
    total_files = counter.most_common(1)[0][1] - counter[sub]
    ts_per_file = math.ceil(total_files / counter[sub])
    assert ts_per_file <= len(trans), f"not enough files in {sub}"
    print(f"Adding {total_files} to category {sub}")
    sub = os.path.join(args.file, sub)
    for f in n_randchoices(listdir_files(sub), total_files):
        random.shuffle(trans)
        transform_file(
            os.path.join(sub, f),
            args.out,
            trans[0 : min(ts_per_file, total_files)],
            args.save,
        )
        total_files -= ts_per_file
        if total_files < 1:
            break


def transform_file(file, dir, ts, save):
    img = imread(file)
    images = {"Original": img}
    for t in ts:
        images[t.__doc__] = t(img)
    if save:
        return save_images(images, dir, file)
    return images


def save_images(images: dict, dir: str, file: str):
    if not dir:
        dir = os.path.dirname(file)
    file = os.path.basename(file)[:-4]
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for key, img in images.items():
        if key == "Original":
            continue
        f = os.path.join(dir, f"{file}_{key}.JPG")
        imsave(f, img)
    return images


def plot_images(images: dict):
    for i, (key, img) in enumerate(images.items()):
        plt.subplot(1, len(images), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap="grey")
        plt.xlabel(key)

    plt.show()
    # histogram(images["Original"])


if __name__ == "__main__":
    try:
        tranformation()
    except Exception as error:
        print(type(error).__name__ + ":", error)
