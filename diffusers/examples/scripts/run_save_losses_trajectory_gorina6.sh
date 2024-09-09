#! /bin/sh

GPU_ID=1
configuration="--resolution=64 --num_generated_samples=500 --save_frequency=1 --batch_size=2 --guidance_scale=7.5 --verbose=False"
analyzed_data="--data_file_path=../data/laion2b-ch-data/00001.tar"
save_data="--data_save_file_path=./debug_model_v1_1_nonmember_data_with_attr_laion_ch_part_01.joblib"
model="--model_name=CompVis/stable-diffusion-v1-1"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u text_to_image/save_losses_trajectory.py ${configuration} ${analyzed_data} ${save_data} ${model}
