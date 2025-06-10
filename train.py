#!python

import os
from os.path import join, basename, dirname
import shutil
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras.applications import EfficientNetV2B0  # type: ignore
from keras.utils import image_dataset_from_directory  # type: ignore
from keras.layers import GlobalAveragePooling2D, Dense, Dropout  # type: ignore

from src.utils import listdir_dirs, listdir_files

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def pars_args():
    parser = argparse.ArgumentParser(
        prog="Training", description="Leaffliction Training program"
    )
    parser.add_argument("in_folder")
    args = parser.parse_args()
    return args.in_folder


def load_sets(in_folder: str, out_folder: str, test_split: float):
    os.mkdir(out_folder)
    os.mkdir(join(out_folder, "train"))
    os.mkdir(join(out_folder, "test"))
    files = []
    for sub in listdir_dirs(in_folder):
        os.mkdir(join(out_folder, "train", sub))
        os.mkdir(join(out_folder, "test", sub))
        for file in listdir_files(join(in_folder, sub)):
            files.append(join(in_folder, sub, file))
    files = np.random.permutation(files)
    split = int(len(files) * (1 - test_split))
    for file in files[:split]:
        class_name = basename(dirname(file))
        shutil.copyfile(
            file, join(out_folder, "train", class_name, basename(file))
        )
    for file in files[split:]:
        class_name = basename(dirname(file))
        shutil.copyfile(
            file, join(out_folder, "test", class_name, basename(file))
        )


def main():
    in_folder = pars_args()
    out_folder = join(dirname(in_folder), "leaf_out")
    load_sets(in_folder, out_folder, 0.1)

    train_ds, val_ds = image_dataset_from_directory(
        in_folder,
        shuffle=True,
        image_size=(256, 256),
        validation_split=0.2,
        seed=42,
        batch_size=256,
        subset="both",
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    model, base_model = build_model(len(class_names))
    base_model.trainable = False
    print("Refining")
    train_model(model, 1e-3, train_ds, val_ds)
    base_model.trainable = True
    print("Fine-tuning")
    train_model(model, 1e-5, train_ds, val_ds)

    save_zip(model, class_names, in_folder, out_folder)


def build_model(class_count):
    base_model = EfficientNetV2B0(include_top=False)
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(class_count, activation="softmax"),
        ]
    )
    return model, base_model


def train_model(model: Sequential, learning_rate: float, train_ds, val_ds):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        shuffle=True,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )


def save_zip(model, class_names, in_folder, out_folder):
    model.save(join(out_folder, "model.keras"))
    with open(join(out_folder, "class_names.json"), "w") as outfile:
        json.dump(class_names, outfile)
    shutil.make_archive(
        join(dirname(in_folder), "Leaffliction"),
        format="zip",
        root_dir=out_folder,
    )
    shutil.rmtree(out_folder)
    print("Archive created:", join(dirname(in_folder), "Leaffliction.zip"))


def plot(history):
    plt.subplot(211)
    plt.title("Validation Loss")
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.subplot(212)
    plt.title("Validation Accuracy")
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ":", error)
