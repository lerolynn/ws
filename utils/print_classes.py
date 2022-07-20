from PIL import Image
import numpy as np


# label = np.asarray(Image.open("../../VOC2012/SegmentationClass_gt/2007_000039.png"), dtype=np.int32)
label = np.asarray(Image.open("../../VOC2012/SegmentationClass_pseudo/2007_000063.png"), dtype=np.int32)

label_list = label.flatten().tolist()
print(label_list)