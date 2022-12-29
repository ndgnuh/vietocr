# from argparse import ArgumentParser
from sys import argv
from icecream import ic
from os import path
import re
import uuid


def find_common_root(files):
    dirs = [path.dirname(file) for file in files]
    return path.commonpath(dirs)


def append_common_root(root, file):
    with open(file, "r", encoding="utf-8") as f:
        annotations = [path.join(root, line.strip())
                       for line in f.readlines()]
    return "\n".join(annotations)


def main():
    files = argv[1:]
    assert len(files) > 1, "Must have at least 3 inputs (src1, src2, ... target)"
    sources = files

    root = find_common_root(sources)
    annotations = [append_common_root(root, file).strip() for file in sources]
    annotations = "\n".join(annotations)

    target = uuid.uuid5(uuid.NAMESPACE_DNS, annotations)
    target = path.join(root,  f"{target}.txt")

    with open(target, "w", encoding="utf-8") as f:
        f.write(annotations)
    print(f"Output written to {target}")


if __name__ == "__main__":
    main()
