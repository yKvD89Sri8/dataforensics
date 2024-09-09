#! /bin/sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=29:59:59
#SBATCH --job-name=jgeng_diffusion_analysis
#SBATCH --output=jgeng_diffusion_analysis_output_log.out

#uenv verbose cuda-11.4 cudnn-11.4-8.2.4
#conda activate base
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate base

#GPU_ID=3
NUM_OF_TRAINING_SAMPLES=40000
MODEL_TRAINING_STEPS=890000
DATA_TYPE="member_data"
DS_NAME="cifar10"
DIFFUSION_STEPS=4000
MODEL_PATH="models/${DS_NAME}_${NUM_OF_TRAINING_SAMPLES}trainingsamples_usekl_false/model${MODEL_TRAINING_STEPS}.pt"
#DATA_DIR="datasets/celeba_gender_4096trainingsamples_290000trainingsteps_40000generatingsamples"
DATA_DIR="../datasets/${DS_NAME}_${NUM_OF_TRAINING_SAMPLES}_partition/target_train"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

mkdir tmp/all_metrics_loss_analysis_"${DS_NAME}"_"${NUM_OF_TRAINING_SAMPLES}"samples_"${TIMESTAMP}"
SAVE_PATH="tmp/all_metrics_loss_analysis_${DS_NAME}_${NUM_OF_TRAINING_SAMPLES}samples_${TIMESTAMP}"
ANALYSIS_DATA_SAVE_PATH="${SAVE_PATH}/${NUM_OF_TRAINING_SAMPLES}trainingsamples_${DATA_TYPE}"

MODEL_FLAGS="--image_size=64 --num_channels=128 --num_res_blocks=3 --learn_sigma=True --dropout=0.3 --model_path=${MODEL_PATH}"
DIFFUSION_FLAGS="--diffusion_steps=${DIFFUSION_STEPS} --log_interval=512 --noise_schedule=linear"
TRAIN_FLAGS="--batch_size=512"
DATA_FLAGS="--num_samples=${NUM_OF_TRAINING_SAMPLES} --dataset_name=${DS_NAME} --data_dir=${DATA_DIR} --analysis_data_save_path=${ANALYSIS_DATA_SAVE_PATH}"

echo DIFFUSION_FLAG=$DIFFUSION_FLAGS >> "$SAVE_PATH"/config_info.txt
echo TRAIN_FLAGS=$TRAIN_FLAGS >> "$SAVE_PATH"/config_info.txt
echo MODEL_FLAGS=$MODEL_FLAGS >> "$SAVE_PATH"/config_info.txt
echo DATA_FLAGS=$DATA_FLAGS >> "$SAVE_PATH"/config_info.txt
echo LOG_SAVE_PATH=tmp/exp_$TIMESTAMP >> "$SAVE_PATH"/config_info.txt

python -u scripts/image_vlb_loss_analysis.py ${DATA_FLAGS} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} 
