from .voc import VOC, VOCAug, pseudoVOC
from .coco import COCO, pseudoCOCO
from .cocostuff import CocoStuff10k, CocoStuff164k


def get_dataset(name):
    return {
        "coco": COCO,
        "pseudococo": pseudoCOCO,
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "pseudovoc": pseudoVOC,
        "vocaug": VOCAug,
    }[name]
