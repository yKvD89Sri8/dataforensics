#! /bin/sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=23:59:59
#SBATCH --job-name=jgeng_diffusion_analysis
#SBATCH --output=jgeng_diffusion_analysis_output_log.out

#uenv verbose cuda-11.4 cudnn-11.4-8.2.4
#conda activate base
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate base

configuration="--resolution=64 --num_generated_samples=500 --save_frequency=1 --batch_size=15 --guidance_scale=0"
analyzed_data="--data_file_path=../data/laion2b-ch-data/00001.tar"
save_data="--data_save_file_path=./model_v1_1_nonmember_data_with_attr_laion_ch_part_01_woguidance.joblib"
model="--model_name=CompVis/stable-diffusion-v1-1"

python -u text_to_image/save_losses_trajectory.py ${configuration} ${analyzed_data} ${save_data} ${model}
