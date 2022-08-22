#!/bin/bash

# Download Pascal VOC2012
wget -P ./data http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_11-May-2012.tar ./data

mv ./data/VOCdevkit/VOC2012/ ./data/VOC2012

rm -r ./data/VOCdevkit/ ./data/VOCtrainval_11-May-2012.tar
# -------------------------