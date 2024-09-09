#! /bin/sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=23:59:59
#SBATCH --job-name=jgeng_diffusion_analysis
#SBATCH --output=jgeng_diffusion_analysis_output_log.out

#conda init bash
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate base

#GPU_ID=0
NUM_TRAIN_DATA=5000
TRAINNING_STEP=290000
DS_NAME="cifar10"
#RESUME_CHECKPOINT="models/${DS_NAME}_${NUM_TRAIN_DATA}trainingsamples_usekl_false/model${TRAINNING_STEP}.pt"
RESUME_CHECKPOINT=""
MODEL_FLAGS="--image_size=64 --num_channels=128 --num_res_blocks=3 --learn_sigma=True --dropout=0.3 --use_kl=False"
DIFFUSION_FLAGS="--diffusion_steps=4000 --noise_schedule=linear"
TRAIN_FLAGS="--lr=1e-4 --batch_size=10 --dataset_name=${DS_NAME} --num_temp_samples=8 --num_train_data=${NUM_TRAIN_DATA}"
DATA_FLAGS="--data_dir=../guided-diffusion/datasets/${DS_NAME}_${NUM_TRAIN_DATA}_partition/target_train --resume_checkpoint=${RESUME_CHECKPOINT}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir tmp/train_"${NUM_TRAIN_DATA}"_"${TIMESTAMP}"

SAVE_PATH=tmp/train_"${NUM_TRAIN_DATA}"_"${TIMESTAMP}"

echo DIFFUSION_FLAG=$DIFFUSION_FLAGS >> "${SAVE_PATH}"/config_info.txt
echo TRAIN_FLAGS=$TRAIN_FLAGS >> "${SAVE_PATH}"/config_info.txt
echo MODEL_FLAGS=$MODEL_FLAGS >> "${SAVE_PATH}"/config_info.txt
echo DATA_FLAGS=$DATA_FLAGS >> "${SAVE_PATH}"/config_info.txt
echo LOG_SAVE_PATH=${SAVE_PATH} >> "${SAVE_PATH}"/config_info.txt

#CUDA_VISIBLE_DEVICES=${GPU_ID} 
python -u scripts/image_train.py ${DATA_FLAGS} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} 
#> "${SAVE_PATH}"/log.txt
