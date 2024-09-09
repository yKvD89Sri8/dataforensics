#! /bin/sh

GPU_ID=1
configuration="--task_type=member --resolution=512 --num_generated_samples=500 --save_frequency=1 --batch_size=2 --guidance_scale=0.75 --image_type=gray --noverbose=False"
data_file="--data_file_path=./size_original/full_reconstruct_model_v1_4_member_data_laion_aesthetic_512relax_part_1040_rgb.joblib"
save_data="--data_save_file_path=./intermediate_image_member"
model="--model_name=CompVis/stable-diffusion-v1-4"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u text_to_image/save_intermediate_generation_images.py ${configuration} ${data_file} ${save_data} ${model}
