import os
import random


def listdir_dirs(dir: str):
    assert os.path.isdir(dir), f"{dir} is not a folder"
    return [s for s in os.listdir(dir) if os.path.isdir(os.path.join(dir, s))]


def listdir_files(dir: str):
    assert os.path.isdir(dir), f"{dir} is not a folder"
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


def n_randchoices(choices: list, count: int) -> list:
    choices = choices.copy()
    random.shuffle(choices)
    return choices[0:count]
