export CUDA_VISIBLE_DEVICES='0,1,2,3'
#EXP_NAME="SSD300_full_precision_teacher"
python train.py #--dataset 'COCO' --dataset_root '/media/apple/Datasets/coco/' #--exp_name $EXP_NAME

