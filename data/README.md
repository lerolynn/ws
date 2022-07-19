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