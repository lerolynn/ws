
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

def get_strided_size(orig_size, stride):
    return (torch.div(orig_size[0]-1, stride, rounding_mode='trunc') + 1, 
            torch.div(orig_size[1]-1, stride, rounding_mode='trunc') + 1)


dataset = VOCSemanticSegmentationDataset(split="train", data_dir="../data/VOC2012")
labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

with open("voc12/train.txt") as f:
    ids = [x.strip() for x in f.readlines()]

for i, id in enumerate(tqdm(ids)):
    # orig = np.load(os.path.join("result/voc12/cam", id + '.npy'), allow_pickle=True).item()

    # cam_dict = np.load(os.path.join("result/voc12/crf/dict", id + '.npy'), allow_pickle=True).item()
    # keys = list(cam_dict.keys())
    # keys = np.array([keys[i] for i in range(len(keys))])
    # high_res_cams = np.stack(list(cam_dict.values()), axis=0)

    # high_res_shape = high_res_cams.shape
    # cam_shape = (int(torch.div(high_res_shape[2], 4, rounding_mode='trunc') + 1),
    #             int(torch.div(high_res_shape[1], 4, rounding_mode='trunc') + 1))
    # cams = list(cam_dict.values())
    # cams = np.stack([cv2.resize(cams[i], dsize=cam_shape, interpolation=cv2.INTER_LINEAR) for i in range(len(cams))], axis=0)
    # cams = torch.from_numpy(cams)

    cam_dict = np.load(os.path.join("result/voc12/crf/10", id + '.npy'), allow_pickle=True).item()
    keys = list(cam_dict.keys())[1:]
    keys = np.array([keys[i]-1 for i in range(len(keys))])
    high_res_cams = np.stack(list(cam_dict.values())[1:], axis=0)

    high_res_shape = high_res_cams.shape
    cam_shape = (int(torch.div(high_res_shape[2]-1, 4, rounding_mode='trunc') + 1),
                int(torch.div(high_res_shape[1]-1, 4, rounding_mode='trunc') + 1))
    cams = list(cam_dict.values())[1:]
    cams = np.stack([cv2.resize(cams[i], dsize=cam_shape, interpolation=cv2.INTER_LINEAR) for i in range(len(cams))], axis=0)
    cams = torch.from_numpy(cams)

    np.save(os.path.join("result/voc12/cg_cam", id + '.npy'),
            {"keys": keys, "cam": cams, "high_res": high_res_cams})


preds = []

dataset = VOCSemanticSegmentationDataset(split="train", data_dir="../data/VOC2012")
labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

for i, id in enumerate(tqdm(dataset.ids)):

    orig = np.load(os.path.join("result/voc12/cg_cam", id + '.npy'), allow_pickle=True).item()
    keys = np.pad(orig['keys'] + 1, (1, 0), mode='constant')
    cams = orig['high_res']


    # cam_dict = np.load(os.path.join("result/voc12/06", id + '.npy'), allow_pickle=True).item()
    # keys = np.array(list(cam_dict.keys()))
    # cams = np.stack(list(cam_dict.values())[1:], axis=0)

    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.15)
    cls_labels = np.argmax(cams, axis=0)
    cls_labels = keys[cls_labels]
    preds.append(cls_labels.copy())

confusion = calc_semantic_segmentation_confusion(preds, labels)
gtj = confusion.sum(axis=1)
resj = confusion.sum(axis=0)
gtjresj = np.diag(confusion)
denominator = gtj + resj - gtjresj
iou = gtjresj / denominator

print({'iou': iou, 'miou': np.nanmean(iou)})