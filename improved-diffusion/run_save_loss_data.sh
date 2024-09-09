#! /bin/sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=23:59:59
#SBATCH --job-name=jgeng_diffusion_analysis
#SBATCH --output=jgeng_diffusion_analysis_output_log.out

#conda activate base
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate base

#GPU_ID=2
NUM_OF_TRAINING_SAMPLES=4096
DATA_TYPE="member_data"
DS_NAME="celeba_gender"
MODEL_PATH="models/celeba_gender_4096originaltrainingsample_40000generatingsamples_4000diffusionsteps_training_usekl_False/model210000.pt"
#DATA_DIR="datasets/celeba_gender_4096trainingsamples_290000trainingsteps_40000generatingsamples"
DATA_DIR="../datasets/${DS_NAME}_${NUM_OF_TRAINING_SAMPLES}_partition/target_train"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

mkdir tmp/loss_analysis_"${DS_NAME}"_"${NUM_OF_TRAINING_SAMPLES}"samples_"${TIMESTAMP}"
SAVE_PATH="tmp/loss_analysis_${DS_NAME}_${NUM_OF_TRAINING_SAMPLES}samples_${TIMESTAMP}"
ANALYSIS_DATA_SAVE_PATH="${SAVE_PATH}/${NUM_OF_TRAINING_SAMPLES}trainingsamples_${DATA_TYPE}"

MODEL_FLAGS="--image_size=64 --num_channels=128 --num_res_blocks=3 --learn_sigma=True --dropout=0.3 --model_path=${MODEL_PATH}"
DIFFUSION_FLAGS="--diffusion_steps=4000 --log_interval=1000 --noise_schedule=linear"
TRAIN_FLAGS="--batch_size=100"
DATA_FLAGS="--num_samples=${NUM_OF_TRAINING_SAMPLES} --dataset_name=${DS_NAME} --data_dir=${DATA_DIR} --analysis_data_save_path=${ANALYSIS_DATA_SAVE_PATH}"

echo DIFFUSION_FLAG=$DIFFUSION_FLAGS >> "$SAVE_PATH"/config_info.txt
echo TRAIN_FLAGS=$TRAIN_FLAGS >> "$SAVE_PATH"/config_info.txt
echo MODEL_FLAGS=$MODEL_FLAGS >> "$SAVE_PATH"/config_info.txt
echo DATA_FLAGS=$DATA_FLAGS >> "$SAVE_PATH"/config_info.txt
echo LOG_SAVE_PATH=tmp/exp_$TIMESTAMP >> "$SAVE_PATH"/config_info.txt

#CUDA_VISIBLE_DEVICES=${GPU_ID} 
python -u scripts/save_false_positive_rate.py ${DATA_FLAGS} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} 
#> "$SAVE_PATH"/log.txt
