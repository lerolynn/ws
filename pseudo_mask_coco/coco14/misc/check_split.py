
from pycocotools.coco import COCO
annFile = "../../../data/coco2014/Annotations/instances_val2014.json"

coco = COCO(annFile)
ids = sorted(list(coco.imgs.keys()))
print("ids ", len(ids))

subdir = "val2014"
file_object = open((subdir + '.txt'), 'r')

filenames = []
for line in file_object:
    filenames.append(int(line.strip()))

different = 0
for i in range(len(filenames)):
    if int(filenames[i]) != int(ids[i]):
        different += 1
        print(filenames[i], ids[i])

print("NOTSAME",  different)