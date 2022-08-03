# Data 

## VOC

Folders

```
|
|- Annotations
|- ImageSets
|- JPEGImages
|- SegmentationClass
|- SegmentationClass_gt
|- SegmentationClass_pseudo
|- SegmentationClassAug
|- SegmentationObject

```
### USED data

SegmentationClass
- Original pixel-level ground truth labels in color
- Labels use the original Pascal VOC colormap
- `trainval.txt` split, 2913 items

SegmentationClass_gt
- Ground truth labels using single color channel
- Converted using `utils/remove_colormap.py` from SegmentationClass labels
- `trainval.txt` split, 2913 items

SegmentationClass_pseudo
- pseudo labels generated from IRN
- `train.txt` split, 1464 items


### Others


## COCO

```
|
|- Annotations
|- coco_seg_anno
|- JPEGImages
| | - train2014
| | - val2014
 - labels.txt

```

Annotations and JPEGImages are downloaded from the COCO official website

`coco_seg_anno` has the image annotations for semantic segmentation. The annotations can be downloaded from the link below and is provide by the [RIB](https://github.com/jbeomlee93/RIB) repository

```
gdown https://drive.google.com/uc?id=1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV
```

