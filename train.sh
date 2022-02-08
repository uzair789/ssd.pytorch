export CUDA_VISIBLE_DEVICES='0,1,2,3'
EXP_NAME="teacher_dummy"
python train.py --dataset 'COCO' --dataset_root '/media/apple/Datasets/coco/' #--exp_name $EXP_NAME

