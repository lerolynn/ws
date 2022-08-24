from .voc import VOC, VOCAug, pseudoVOC
from .coco import COCO, pseudoCOCO


def get_dataset(name):
    return {
        "coco": COCO,
        "pseudococo": pseudoCOCO,
        "voc": VOC,
        "pseudovoc": pseudoVOC,
        "vocaug": VOCAug,
    }[name]
