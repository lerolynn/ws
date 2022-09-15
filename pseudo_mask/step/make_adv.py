import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import cv2

import voc12.dataloader
import coco14.dataloader

from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, args, thres):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # Save CAM as image
            raw_img = np.asarray(cv2.imread(os.path.join("../data/VOC2012/JPEGImages",img_name+".jpg")))
            cam_map, _ = torch.max(highres_cam, dim=0)
            cam_map = cam_map.cpu().numpy()

            erase_mask = np.zeros((cam_map.shape))
            erase_mask[cam_map > thres] = 1
            idx = (erase_mask==1)

            cam_map = plt.cm.jet_r(cam_map)[..., :3] * 255.0
            cam_output = (cam_map.astype(np.float) * (1/3) + raw_img.astype(np.float) * (2/3))
            # Save cam images
            # outfile = os.path.join("result/voc12/cam_img", img_name + ".png")
            # cv2.imwrite(outfile, cam_output)

            # Apply mask to raw image
            raw_img[idx] = 0
            cam_output = raw_img.astype(np.float)

            # Save erased images
            outfile = os.path.join("result/voc12/erased_jpg", img_name + ".png")
            cv2.imwrite(outfile, cam_output)

            # save cams
            np.save(os.path.join(args.prev_cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args, thres):
    if args.coco:
        model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=80)
    else:
        model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    if args.coco:
        dataset = coco14.dataloader.COCO14ClassificationDatasetMSF(args.train_list,
                                                             coco14_root=args.coco14_root, scales=args.cam_scales)
        
    else:
        dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args, thres), join=True)
    print(']')

    torch.cuda.empty_cache()