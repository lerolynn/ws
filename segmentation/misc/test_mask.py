import imageio
import cv2
import numpy as np

# imageio.imread()

pseudo_path = "../../data/VOC2012/SegmentationClass_pseudo/2007_000039.png"
gt_path = "../../data/VOC2012/SegmentationClass_gt/2007_000039.png"


pseudo = cv2.imread(pseudo_path, cv2.IMREAD_COLOR).astype(np.float32)
gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32)

print(pseudo.shape, gt.shape)
print(set(pseudo.flatten()), set(gt.flatten()))