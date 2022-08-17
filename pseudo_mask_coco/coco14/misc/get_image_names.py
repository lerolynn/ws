"""
get_image_names.py

This file reads the filenames of the image datasets in the specified folders
and generates txt files for the labels

"""

from unicodedata import numeric
import numpy as np
import glob
import os

subdir = "val2014"
dir_name = os.path.join("../../../data/coco2014", subdir)

open((subdir + '.txt'), 'w').close()
file_object = open((subdir + '.txt'), 'a')

filenames = []
for path in os.listdir(dir_name):
    filenames.append(path.split(".")[0].split("_")[-1])

filenames.sort()

for filename in filenames:
    file_object.write(str(filename + "\n"))

file_object.close()