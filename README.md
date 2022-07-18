# Weakly Supervised Semantic Segmentation

## Usage Instructions

### IRN

in `pseudo_mask` directory:

```python
conda activate ws
cd pseudo_mask
```

```python
python run_sample.py --voc12_root ../data/VOC2012
```


### Segmentation

In `segmentation` directory:

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
python main.py crf --config-path configs/voc12.yaml
```
