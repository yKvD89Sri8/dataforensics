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

configuration="--resolution=512 --num_generated_samples=1000 --save_frequency=1 --batch_size=10 --guidance_scale=7.5 --image_type=rgb --noverbose=True --ds_name=mscoco  "
analyzed_data="--data_file_path=../data/mscoco/{00000..00056}.tar"
save_data="--data_save_file_path=./size_original/full_reconstruct_model_v1_4_nonmember_data_mscoco_data_part_0056_0_rgb.joblib"
model="--model_name=CompVis/stable-diffusion-v1-4"

python -u text_to_image/save_losses_trajectory_from_full_reconstructed.py ${configuration} ${analyzed_data} ${save_data} ${model}
