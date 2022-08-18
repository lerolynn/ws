import random

dataset_size = 2000

with open('../data/coco2014/ImageSets/train2014_orig.txt') as f:
    train_lines = f.readlines()

train_shuffled = random.sample(train_lines, int(0.9 * dataset_size))
train_shuffled.sort()
print(train_shuffled)

with open('../data/coco2014/ImageSets/train2014.txt', 'w') as f:
    for line in train_shuffled:
        f.write(line)

with open('../data/coco2014/ImageSets/val2014_orig.txt') as f:
    lines = f.readlines()

shuffled = random.sample(lines, int(0.1 * dataset_size))
shuffled.sort()
print(shuffled)

with open('../data/coco2014/ImageSets/val2014.txt', 'w') as f:
    for line in shuffled:
        f.write(line)