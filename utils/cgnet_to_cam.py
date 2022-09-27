
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

from PIL import Image
from tqdm import tqdm

dataset = VOCSemanticSegmentationDataset(split="train", data_dir="../data/VOC2012")
labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

for i, id in enumerate(tqdm(dataset.ids)):
    cam_dict = np.load(os.path.join("result/voc12/crf/06", id + '.npy'), allow_pickle=True).item()
    keys = list(cam_dict.keys())[1:]
    keys = np.array([keys[i]-1 for i in range(len(keys))])
    cams = np.stack(list(cam_dict.values())[1:], axis=0)

    np.save(os.path.join("result/voc12/cg_cam", id + '.npy'),
            {"keys": keys, "high_res": cams})


# preds = []

# dataset = VOCSemanticSegmentationDataset(split="train", data_dir="../data/VOC2012")
# labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

# for i, id in enumerate(tqdm(dataset.ids)):

#     orig = np.load(os.path.join("result/voc12/cam", id + '.npy'), allow_pickle=True).item()
#     keys = np.pad(orig['keys'] + 1, (1, 0), mode='constant')
#     cams = orig['high_res']


#     # cam_dict = np.load(os.path.join("result/voc12/06", id + '.npy'), allow_pickle=True).item()
#     # keys = np.array(list(cam_dict.keys()))
#     # cams = np.stack(list(cam_dict.values())[1:], axis=0)

#     cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.15)
#     cls_labels = np.argmax(cams, axis=0)
#     cls_labels = keys[cls_labels]
#     preds.append(cls_labels.copy())

# confusion = calc_semantic_segmentation_confusion(preds, labels)
# gtj = confusion.sum(axis=1)
# resj = confusion.sum(axis=0)
# gtjresj = np.diag(confusion)
# denominator = gtj + resj - gtjresj
# iou = gtjresj / denominator

# print({'iou': iou, 'miou': np.nanmean(iou)})