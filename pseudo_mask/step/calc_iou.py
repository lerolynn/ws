
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from collections import Counter

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def run():
    preds = []
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    print(thresholds)
    dataset = VOCSemanticSegmentationDataset(split="train", data_dir="../data/VOC2012")
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    
    best_threshold = np.zeros(len(dataset))
    best_iou = np.zeros(len(dataset))

    for i, id in enumerate(tqdm(dataset.ids)):
        cam_dict = np.load(os.path.join("result/voc12/cam", id + '.npy'), allow_pickle=True).item()
        orig_cams = cam_dict['high_res']
        scores = np.zeros(len(thresholds))
        for j, threshold in enumerate(thresholds):
            cams = np.pad(orig_cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)

            pred_label = keys[cls_labels]
            gt_label = labels[i]

            # Ignore pixels where label is -1
            ignore_mask = gt_label == -1
            gt_label[ignore_mask] = 0
            pred_label[ignore_mask] = 0

            # Get IoU of label
            intersection = np.logical_and(pred_label, gt_label)
            union = np.logical_or(pred_label, gt_label)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(id, round(threshold, 2), "\t\tiou score:", iou_score)
            scores[j] = round(iou_score, 4)

        idx = np.argmax(scores)
        best_threshold[i] = round(thresholds[idx],2)
        best_iou[i] = scores[idx]

    print(Counter(list(best_threshold)))
    print("IoU", np.average(best_iou))
    
    # plt.hist(best_threshold, width=0.04, align="mid")
    # plt.xticks(thresholds)
    # plt.show()

run()