import numpy as np

labels =np.load("../RIB_split/cls_labels_coco.npy", allow_pickle=True).item()

print(np.where(labels[25] == 1))