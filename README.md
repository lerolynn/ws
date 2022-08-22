# Weakly Supervised Semantic Segmentation

This repository runs the full Weakly Supervised Semantic Segmentaiton pipeline based 

## Installation

This repository is tested on Ubuntu 18.04, using Python 3.7 and Pytorch >= 1.11 Other environment variables are as specified in the `environment.yml` file.

Run the following to create the conda environment to run the repository.

```console
conda env create -f environment.yml
```

_Note: Pytorch environment should be suitable to your CUDA version. This repository was tested on Pytorch versions 1.11.0 and 1.12.0 for the Nvidia RTX 2080Ti and 3090 GPUs respectively._

## Usage

### Dataset

In the root directory, run `setup.sh` to to download the PASCAL VOC2012 and the MS COCO2014 datasets and setup the directory hierarchy as specified [here](./data/README.md)

```console
./setup.sh
```

#### PASCAL VOC2012

[Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and extract the Pascal VOC2012 dataset from the [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).

```console
cd data

wget -P data http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar ; mv VOCdevkit/VOC2012/ VOC2012 ; rm -r VOCdevkit/ VOCtrainval_11-May-2012.tar
```

#### MS COCO2014

Download and extract the MS COCO dataset from the offical [COCO website](https://cocodataset.org/#download).

```console
cd data/coco

wget http://images.cocodataset.org/zips/train2014.zip http://images.cocodataset.org/zips/val2014.zip http://images.cocodataset.org/annotations/annotations_trainval2014.zip ; unzip train2014.zip ; unzip val2014.zip ; unzip annotations_trainval2014.zip ; rm train2014.zip val2014.zip annotations_trainval2014.zip

mkdir JPEGImages ; mv train2014 JPEGImages/train2014 ; mv val2014 JPEGImages/val2014 ; mv annotations Annotations

```

_Further information on the dataset and the data root folder is located [here](./data/README.md)._

## Pretrained Weights

Download the pretrained weights for segmentation
In `segmentation` directory:

To download weights pretrained on imagenet for segmentation
```
cd segmentation/data/models/imagenet
gdown https://drive.google.com/uc?id=14soMKDnIZ_crXQTlol9sNHVPozcQQpMn
```


## Usage

Activate the environment variables
```
conda activate wsss
```

### IRN (VOC2012 dataset)

In `pseudo_mask` directory:

```python

cd pseudo_mask
```

```python
python run_sample.py --voc12_root ../data/VOC2012
```

```python
python run_sample.py --voc12_root ../data/VOC2012 --infer_list voc12/val.txt --train_cam_pass False \
--make_cam_pass False --eval_cam_pass False --cam_to_ir_label_pass False \
--train_irn_pass False --make_ins_seg_pass False --eval_ins_seg_pass False --make_sem_seg_pass False
```

In home directory:
```
cp -r pseudo_mask/result/sem_seg data/VOC2012/SegmentationClass_pseudo
```

### IRN (COCO dataset)

#### setup

```
pip install pycocotools
```

In `pseudo_mask_coco` directory:

```python
python run_sample.py

python run_sample.py \
--make_cam_pass False --eval_cam_pass False --cam_to_ir_label_pass False --train_irn_pass False \
--make_ins_seg_pass False --eval_ins_seg_pass False --make_sem_seg_pass False --eval_sem_seg_pass False

```

### Segmentation

```console
conda activate deeplab-pytorch
cd segmentation
```

Train Deeplab v2 on Pascal VOC2012
```console
python main.py train --config-path configs/voc12.yaml

python main.py train --config-path configs/coco.yaml
```

Evaluate performance on validation set

```console
python main.py test --config-path configs/voc12.yaml --model-path data/models/voc12/deeplabv2_resnet101_msc/train/checkpoint_final.pth

python main.py test --config-path configs/coco.yaml --model-path output/coco/models/coco/deeplabv2_resnet101_msc/train2014/checkpoint_final.pth
```

Evaluate with CRF post-processing
```console
python main.py crf --config-path configs/coco.yaml
```

## Acknowledgment