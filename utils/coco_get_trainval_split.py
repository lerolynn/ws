"""
get_image_names.py

This file reads the filenames of the image datasets in the specified folders
and generates txt files for the labels

"""

from unicodedata import numeric
import numpy as np
import os

def work(subdir):
    dir_name = os.path.join("./data/coco2014/JPEGImages", subdir)
    
    write_file = os.path.join("./data/coco2014/ImageSets", (subdir + '.txt'))

    open(write_file, 'w').close()

    file_object = open(write_file, 'a')

    filenames = []
    for path in os.listdir(dir_name):
        filenames.append(path.split(".")[0].split("_")[-1])

    filenames.sort()
    print(len(filenames))

    for filename in filenames:
        file_object.write(str(filename + "\n"))

    file_object.close()

if __name__ == '__main__':
    
    train_dir = "train2014"
    work(train_dir)

    val_dir = "val2014" 
    work(val_dir)