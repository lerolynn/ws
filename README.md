# Weakly Supervised Semantic Segmentation

## Installation

This repository is tested on Ubuntu 18.04, using Python 3.7 and Pytorch 1.11 Other environment variables are specified in the `environment.yml` file.

Run the following to create the conda environment to run the repository.

```console
conda env create -f environment.yml
```

## Dataset


## Usage

### IRN (VOC2012 dataset)

In `pseudo_mask` directory:

```python
conda activate wsss
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

In `segmentation` directory:

To download weights pretrained on imagenet for segmentation
```
cd segmentation/data/models/imagenet
gdown https://drive.google.com/uc?id=14soMKDnIZ_crXQTlol9sNHVPozcQQpMn
```

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