#!/bin/bash

conda env create -f environment.yml
conda activate wsss

# Download Pascal VOC2012 -----------------------
wget -P ./data http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar -xvf ./data/VOCtrainval_11-May-2012.tar -C ./data

mv ./data/VOCdevkit/VOC2012/ ./data/VOC2012

rm -r ./data/VOCdevkit/ ./data/VOCtrainval_11-May-2012.tar
# -----------------------------------------------

# Download MS COCO ------------------------------
mkdir -p data/coco2014/JPEGImages

# Download COCO training data
wget -P ./data/coco2014/JPEGImages http://images.cocodataset.org/zips/train2014.zip 
unzip data/coco2014/JPEGImages/train2014.zip -d data/coco2014/JPEGImages
rm data/coco2014/JPEGImages/train2014.zip 

# Download COCO validation data
wget -P ./data/coco2014/JPEGImages http://images.cocodataset.org/zips/val2014.zip 
unzip data/coco2014/JPEGImages/val2014.zip -d data/coco2014/JPEGImages
rm data/coco2014/JPEGImages/val2014.zip 

# Download COCO annotations
wget -P ./data/coco2014 http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip data/coco2014/annotations_trainval2014.zip -d data/coco2014
rm data/coco2014/annotations_trainval2014.zip 
mv ./data/coco2014/annotations ./data/coco2014/Annotations
# ----------------------------------------------