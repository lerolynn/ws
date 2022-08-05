import numpy as np

labels =np.load("../cls_labels_coco.npy", allow_pickle=True).item()

print(labels[2011003271].shape)