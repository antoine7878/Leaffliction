#!python

import os
import math
import random
import argparse
import warnings
import concurrent.futures

import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from Distribution import count_class
from src.augment import AUGMENTATIONS
from src.utils import listdir_dirs, listdir_files, n_randchoices

warnings.filterwarnings("ignore", message="This might be a color image")


def pars_args():
    parser = argparse.ArgumentParser(
        prog="Augmentation", description="Leaffliction Augmentation program"
    )
    parser.add_argument("file")
    parser.add_argument("-o", "--out", type=str)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    return args


def augmentation():
    args = pars_args()
    if args.clean:
        clean_folder(args)
    elif os.path.isfile(args.file):
        assert args.file.lower().endswith(".jpg"), f"{args.file}: wrong format"
        images = augment_file(args.file, args.out, AUGMENTATIONS)
        plot_images(images)
    elif os.path.isdir(args.file):
        augment_folder(args)
    else:
        raise TypeError(f"{args.file} not found")


def augment_folder(args):
    counter = count_class(args.file)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    for sub in listdir_dirs(args.file):
        pool.submit(augment_subfolder, counter, sub, args)


def augment_subfolder(counter, sub, args):
    trans = AUGMENTATIONS
    total_files = counter.most_common(1)[0][1] - counter[sub]
    ts_per_file = math.ceil(total_files / counter[sub])
    assert ts_per_file <= len(trans), f"not enough files in {sub}"
    print(f"Adding {total_files} to category {sub}")
    sub = os.path.join(args.file, sub)
    for f in n_randchoices(listdir_files(sub), total_files):
        random.shuffle(trans)
        augment_file(
            os.path.join(sub, f),
            args.out,
            trans[0: min(ts_per_file, total_files)],
        )
        total_files -= ts_per_file
        if total_files < 1:
            break


def augment_file(file, dir, ts):
    img = imread(file)
    images = {"Original": img}
    for t in ts:
        images[t.__doc__] = t(img)
    return save_images(images, dir, file)


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


def clean_folder(args):
    files = []
    for sub in listdir_dirs(args.file):
        for file in os.listdir(os.path.join(args.file, sub)):
            if "_" in file:
                files.append(os.path.join(args.file, sub, file))
                print(sub, file)
    if len(files) == 0:
        print("Nothing to delete")
        return
    answer = input("Confirm deletion: y/[n]:")
    if answer == "y":
        for file in files:
            os.remove(file)
        print("Deleted")
    else:
        print("Canceled")


def plot_images(images: dict):
    plt.figure(figsize=(5, 10))
    for i, (key, img) in enumerate(images.items()):
        plt.subplot(4, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel(key)
    plt.show()


if __name__ == "__main__":
    try:
        augmentation()
    except Exception as error:
        print(type(error).__name__ + ":", error)
