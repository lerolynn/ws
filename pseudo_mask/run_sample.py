import argparse
import os

from misc import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # IRN for VOC dataset
    parser.add_argument("--coco", default=False)

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default="../data/VOC2012", type=str)

    # Dataset
    parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="log/voc12_sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/voc12/res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/voc12/res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/voc12/cam", type=str)
    parser.add_argument("--prev_cam_out_dir", default="result/voc12/prev_cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/voc12/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/voc12/sem_seg", type=str)
    # parser.add_argument("--ins_seg_out_dir", default="result/voc12/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    # Train adversarial erasing step
    parser.add_argument("--train_adv_pass", default=True)

    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=False)
    parser.add_argument("--train_irn_pass", default=False)
    # parser.add_argument("--make_ins_seg_pass", default=True)
    # parser.add_argument("--eval_ins_seg_pass", default=True)
    parser.add_argument("--make_sem_seg_pass", default=False)
    parser.add_argument("--eval_sem_seg_pass", default=False)

    args = parser.parse_args()

    # Store coco variable to pass through args object
    args.coco = False
    os.makedirs("log", exist_ok=True)
    os.makedirs("sess/voc12", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.prev_cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    # os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    os.makedirs("result/voc12/combined_cam", exist_ok=True) # CAM image
    os.makedirs("result/voc12/erased_jpg", exist_ok=True) # erased raw image
    os.makedirs("result/voc12/cam_comp", exist_ok=True) # CAM from complementary activations

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))
    print("\nRunning IRN to make pseudo labels for VOC2012 datset\n")

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    # Recursive adversarial erasing
    if args.train_adv_pass is True:
        import step.train_adv
        import step.make_adv
        import step.eval_cam

        thresh = [0.9, 0.8, 0.7]

        for i in range(len(thresh)):
            timer = pyutils.Timer('step.make_adv:')
            step.make_adv.run(args, thresh[i])
            timer = pyutils.Timer('step.train_adv:')
            step.train_adv.run(args)
            timer = pyutils.Timer('step.eval_cam:')
            step.eval_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    # if args.make_ins_seg_pass is True:
    #     import step.make_ins_seg_labels

    #     timer = pyutils.Timer('step.make_ins_seg_labels:')
    #     step.make_ins_seg_labels.run(args)

    # if args.eval_ins_seg_pass is True:
    #     import step.eval_ins_seg

    #     timer = pyutils.Timer('step.eval_ins_seg:')
    #     step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

