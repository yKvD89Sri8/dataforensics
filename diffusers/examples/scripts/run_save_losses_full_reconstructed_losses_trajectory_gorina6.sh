#! /bin/sh

GPU_ID=6
configuration="--resolution=64 --num_generated_samples=500 --save_frequency=1 --batch_size=2 --guidance_scale=0 --image_type=gray"
analyzed_data="--data_file_path=../data/laion2b-en-data/00001.tar"
save_data="--data_save_file_path=./debug_full_reconstruct_model_v1_1_member_data_with_attr_laion2b_en_part_01.joblib"
model="--model_name=CompVis/stable-diffusion-v1-1"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u text_to_image/save_losses_trajectory_from_full_reconstructed.py ${configuration} ${analyzed_data} ${save_data} ${model}
