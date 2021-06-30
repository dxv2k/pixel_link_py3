# TRAIN_DIR=${HOME}/models/pixel_link
DATASET=icdar2015
# DATASET_DIR={IC15_PATH}/ch4_training_images
CKPT_PATH={MODEL_PATH}/model.ckpt-38055

DATASET_DIR=/content/drive/MyDrive/ICDAR2015/ch4_training_images

python train_pixel_link.py \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-3\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=8\            
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
            --max_number_of_steps=100\
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1