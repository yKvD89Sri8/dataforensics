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

configuration="--task_type=member --resolution=512 --num_generated_samples=500 --save_frequency=1 --batch_size=2 --guidance_scale=0.75 --image_type=gray --noverbose=True"
data_file="--data_file_path=./size_original/full_reconstruct_model_v1_4_member_data_laion2b_aesthetic_part_0027_rgb.joblib"
save_data="--data_save_file_path=./intermediate_image_member"
model="--model_name=CompVis/stable-diffusion-v1-4"

python -u text_to_image/save_intermediate_generation_images.py ${configuration} ${data_file} ${save_data} ${model}
