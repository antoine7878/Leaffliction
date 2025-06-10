#!python

import os
import zipfile
import argparse
import json
from pathlib import Path
from os.path import basename, join, dirname

from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from keras.saving import load_model  # type: ignore
from keras.utils import image_dataset_from_directory  # type: ignore

from utils import listdir_files


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Prediction", description="Leaffliction Prediction program"
    )
    parser.add_argument("archive")
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()
    return args.file, args.archive


def main():
    file, archive_name = parse_args()

    with zipfile.ZipFile(archive_name, mode="r") as archive:
        extract_folder = join(dirname(archive_name), Path(archive_name).stem)
        archive.extractall(extract_folder)

    model = load_model(join(extract_folder, "model.keras"))

    with open(join(extract_folder, "class_names.json"), "r") as openfile:
        class_names = json.load(openfile)

    if not file:
        eval_model(extract_folder, model)
    elif os.path.isdir(file):
        preict_dir(file, model, class_names)
    elif os.path.isfile(file):
        predict_file(file, model, class_names)
    else:
        raise TypeError(f"{file} not found")


def predict_file(file, model, class_names):
    original = imread(file)
    pred = model(original.reshape((1, 256, 256, 3)))
    class_index = np.argmax(pred)
    class_pred = class_names[class_index]

    print(os.path.basename(file))
    print(class_pred)

    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.title(f"{basename(file)}\n{class_pred}", loc="center")
    ax = plt.subplot(1, 2, 1)
    ax.axis("off")
    plt.imshow(original)
    ax = plt.subplot(1, 2, 2)
    ax.axis("off")
    plt.imshow(original)
    plt.show()


def preict_dir(dir: str, model, class_names):
    files = []
    imgs = []
    for file in listdir_files(dir):
        imgs.append(imread(os.path.join(dir, file)))
        files.append(file)
    pred = model(np.array(imgs))
    for p, f in zip(pred, files):
        print(f"{f} predicted: {class_names[np.argmax(p)]}")


def eval_model(extract_folder, model):
    ds = image_dataset_from_directory(
        join(extract_folder, "test"),
        shuffle=True,
        image_size=(256, 256),
        seed=42,
        label_mode="categorical",
    )
    hist = model.evaluate(ds)
    print("Loss", hist[0])
    print("Accuracy", hist[1])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ":", error)
