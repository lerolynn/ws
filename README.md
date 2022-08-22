# Weakly Supervised Semantic Segmentation

This repository runs the full Weakly Supervised Semantic Segmentaiton pipeline following the method of [IRNet](hhttps://github.com/jiwoon-ahn/irn)

## Installation

This repository is tested on Ubuntu 18.04, using Python 3.7 and Pytorch >= 1.11 Other environment variables are as specified in the `environment.yml` file.

_Note: Pytorch environment should be suitable to your CUDA version. This repository was tested on Pytorch versions 1.11.0 and 1.12.0 for the Nvidia RTX 2080Ti and 3090 GPUs respectively._

---

### Quick Setup

To setup the environment variables and install the required datasets, run `setup.sh` in the root directory of the repository.

```console
./setup.sh
```

The setup script is configured to set up the Anaconda environment, download the PASCAL VOC2012, the MS COCO2014 datasets and the pretrained segmentation weights and to 

---

#### Setup Steps:

1. Install and setup the Conda environment `wsss`
2. The datasets are downloaded and moved to the folders as specified in the directory hierarchy [here](./data/README.md). More information on the datasets is located in the [README](./data/README.md) of the data folder.

   * Pascal VOC2012 is downloaded from the official [Pascal VOC website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
   * MS COCO dataset is downloaded from the offical [COCO website](https://cocodataset.org/#download)

3. While VOC provides segmentation masks as images, COCO ground truth segmentation masks have to be converted from the annotation files by running:

```
python utils/coco_ann_to_mask.py
```
4. Download weights used for the segmentation model are pretrained on the ImageNet dataset for fair comparison, and are provided by the authors of [RIB](https://github.com/jbeomlee93/RIB).


---

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