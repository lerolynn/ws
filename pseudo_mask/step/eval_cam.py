
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

from PIL import Image
from tqdm import tqdm

def run(args):
    preds = []

    if args.coco:
        ids = open('coco14/train2014.txt').readlines()
        ids = [i.split('\n')[0] for i in ids]
        labels = []
        n_images = 0

        for i, id in enumerate(tqdm(ids)):
            label = np.array(Image.open('../data/coco2014/gt_mask/train2014/%s.png' % id))
            n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            if not ('high_res' in cam_dict):
                preds.append(np.zeros_like(label))
                labels.append(label)
                continue
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(label)
            xx, yy = cls_labels.shape, label.shape
            if xx[0] != yy[0]:
                print(id, xx, yy)

    else:
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
        labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
        
        for i, id in enumerate(tqdm(dataset.ids)):
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
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