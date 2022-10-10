# NEED TO SET
DATASET_ROOT=../data/VOC2012
GPU=0,1,2,3

# 1. train a classification network and compute refined seed
 CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
     --voc12_root ${DATASET_ROOT} \
     --num_workers 8 \
     --cam_eval_thres 0.15 \
     --eval_cam_pass True \
     --cam_to_ir_label_pass True \
     --conf_fg_thres 0.40 \
     --conf_bg_thres 0.05 \
     --cam_out_dir ../pseudo_mask/cgnet/output 


# 2.1. train an attribute manipulation network
 CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
     --voc12_root ${DATASET_ROOT} \
     --num_workers 8 \
     --train_amn_pass True \
     --cam_out_dir ../pseudo_mask/cgnet/output 

# 2.2. generate activation maps and refined seed for boundary refinement
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
    --voc12_root ${DATASET_ROOT} \
    --num_workers 8 \
    --make_amn_cam_pass True \
    --eval_amn_cam_pass True \
    --amn_cam_to_ir_label_pass True \
    --conf_fg_thres 0.45 \
    --conf_bg_thres 0.15 \
    --cam_out_dir ../pseudo_mask/cgnet/output 

# 3.1. train a boundary refinement network (IRN)
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
    --voc12_root ${DATASET_ROOT} \
    --num_workers 8 \
    --train_irn_pass True \
    --cam_out_dir ../pseudo_mask/cgnet/output 

# 3.2. generate the final pseudo-masks
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
    --voc12_root ${DATASET_ROOT} \
    --num_workers 8 \
    --make_sem_seg_pass True \
    --eval_sem_seg_pass True \
    --cam_out_dir ../pseudo_mask/cgnet/output 
