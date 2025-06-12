#!python

import os
import json
import rembg
import zipfile
import argparse
from pathlib import Path
from os.path import basename, join, dirname

import numpy as np
from skimage.io import imread
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
        extract_dir = join(dirname(archive_name), Path(archive_name).stem)
        archive.extractall(extract_dir)

    model = load_model(join(extract_dir, "model.keras"))

    with open(join(extract_dir, "class_names.json"), "r") as openfile:
        class_names = json.load(openfile)

    if not file:
        eval_model(extract_dir, model)
    elif os.path.isdir(file):
        predict_dir(file, model, class_names)
    elif os.path.isfile(file):
        predict_file(file, model, class_names)
    else:
        raise TypeError(f"{file} not found")


def predict_file(file, model, class_names):
    original = imread(file)
    pred = model(original.reshape((1, 256, 256, 3)))
    class_index = np.argmax(pred)
    class_pred = class_names[class_index]

    print('Dirname:', basename(dirname(file)))
    print("Prediction", class_pred)

    fig = plt.figure()
    fig.patch.set_facecolor("#1e1e1e")
    ax = plt.subplot()
    ax.axis("off")
    plt.title(
        f"== Class predicted: {class_pred} ==",
        fontdict={"fontsize": 20, "color": "limegreen"},
    )
    ax = plt.subplot(1, 2, 1)
    rm_ticks()
    plt.imshow(original)
    ax = plt.subplot(1, 2, 2)
    rm_ticks()
    plt.imshow(rembg.remove(original))
    plt.show()


def rm_ticks():
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
    )


def predict_dir(dir: str, model, class_names):
    dirs = []
    imgs = []
    for file in listdir_files(dir):
        path = os.path.join(dir, file)
        imgs.append(imread(path))
        dirs.append(os.path.basename(os.path.dirname(path)))
    pred = model.predict(np.array(imgs), batch_size=64)
    print(pred)
    correct_count = 0
    for p, f in zip(pred, dirs):
        predi = class_names[np.argmax(p)]
        print(f"{f} predicted: {predi}")
        if predi == f:
            correct_count += 1
    print(f"Accuracy {correct_count / len(pred):.4%}")


def eval_model(extract_dir, model):
    ds = image_dataset_from_directory(
        join(extract_dir, "test"),
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
