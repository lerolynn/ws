# Weakly Supervised Semantic Segmentation

## Usage Instructions

### IRN (VOC2012 dataset)

In `pseudo_mask` directory:

```python
conda activate ws
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
gdown https://drive.google.com/uc?id=14soMKDnIZ_crXQTlol9sNHVPozcQQpMn
```

```python
conda activate deeplab-pytorch
cd segmentation
```

Train Deeplab v2 on PascalVOC2012
```python
python main.py train --config-path configs/voc12.yaml
```

Evaluate performance on validation set

```python
python main.py test --config-path configs/voc12.yaml --model-path data/models/voc12/deeplabv2_resnet101_msc/train/checkpoint_final.pth
```

Evaluate with CRF post-processing
```python
python main.py crf --config-path configs/voc12_test.yaml
```


python main.py test --config-path configs/voc12_test.yaml --model-path data/models/voc12/deeplabv2_resnet101_msc/caffemodel/deeplabv2_resnet101_msc-vocaug.pth

