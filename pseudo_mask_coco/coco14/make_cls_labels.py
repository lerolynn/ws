import argparse
import numpy as np
import os
from pycocotools.coco import COCO

# Keys are the coco-annotated labels, from Annotations, values are the compressed classes 
# Ignores classes that only have superclasses
CAT_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
           11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 
           22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 
           35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 
           46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 
           56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 
           67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 
           80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


def work(name_list, ann_file, d):
    with open(name_list) as f:
        lines = f.readlines()
        img_ids = [int(x.strip()) for x in lines]   
    
    coco = COCO(ann_file)

    # Loop through image ids from id text file
    for img_id in img_ids:

        # Get annotations of the current image (may be empty)
        annIds = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annIds)
        image_cats = np.zeros(80)

        for i in range(len(annotations)):
            entity_id = annotations[i]["category_id"]
            # Subtract 1 to ignore background category
            image_cats[CAT_MAP[entity_id]-1] = 1

            # entity = coco.loadCats(entity_id)[0]["name"]
            # print("{}: {}".format(CAT_MAP[entity_id], entity))    
                 
        d[img_id] = image_cats

    return d


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='train2014.txt', type=str)
    parser.add_argument("--val_list", default='val2014.txt', type=str)
    
    parser.add_argument("--out", default='cls_labels_coco.npy', type=str)
    parser.add_argument("--coco14_root", default='../../../data/coco14', type=str)

    args = parser.parse_args()

    ann_file_train = '../../data/coco2014/Annotations/instances_train2014.json'
    ann_file_val = '../../data/coco2014/Annotations/instances_val2014.json'

    d = dict()
    d = work(args.train_list, ann_file_train, d)
    print(len(d), " Training image labels created\n")

    d = work(args.val_list, ann_file_train, d)
    print(len(d), " Training + Validation image labels created\n")

    np.save(args.out, d)