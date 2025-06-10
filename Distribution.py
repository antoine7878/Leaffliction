#!python

import os
import sys
from collections import Counter
import matplotlib.pyplot as plt

from src.utils import listdir_dirs


def main():
    dir_name = parse_args(sys.argv)
    counter = count_class(dir_name)
    plot_classes(counter)
    for key, value in counter.items():
        print(key, value)


def parse_args(argv):
    assert len(argv) == 2, "Wrong argument number"
    dir_name = os.path.join(os.getcwd(), argv[1])
    assert os.path.isdir(dir_name), "Argurment is not a directory"
    return dir_name


def count_class(dir_name):
    counter = Counter()
    for sub in listdir_dirs(dir_name):
        counter[sub] = count_jpg(os.path.join(dir_name, sub))
    return counter


def count_jpg(directory):
    i = 0
    for file in os.listdir(directory):
        if os.path.isfile(file):
            continue
        if not file.lower().endswith(".jpg"):
            continue
        i += 1
    return i


def plot_classes(counter):
    grid_shape = (2, 1)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    histo = plt.subplot2grid(grid_shape, (0, 0))
    histo.bar(counter.keys(), counter.values(), color=colors)

    pie = plt.subplot2grid(grid_shape, (1, 0))
    pie.pie(counter.values(), labels=counter.keys(), autopct="%1.1f%%")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ":", error)
