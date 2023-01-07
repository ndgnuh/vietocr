from os import path, listdir
import sys


class Module(list):
    def __init__(self):
        dir = path.dirname(__file__)
        image_dir = path.join(dir, "data", "testimages")
        images = [path.join(image_dir, image)
                  for image in listdir(image_dir)]
        super().__init__(images)


# some serious black magic
# https://stackoverflow.com/questions/32594450/is-it-possible-to-make-a-module-iterable-in-python
sys.modules[__name__] = Module()
