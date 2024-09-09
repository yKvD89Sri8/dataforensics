#! /bin/sh

GPU_ID=1
configuration="--num_images_per_prompt=4 --ds_name=mscoco --resolution=512 --num_generated_samples=500 --save_frequency=1 --batch_size=2 --guidance_scale=0.75 --image_type=gray --noverbose=False"
data_file="--data_file_path=../data/mscoco/{00000..00030}.tar"
save_data="--data_save_file_path=./intermediate_image_cluster_images"
model="--model_name=CompVis/stable-diffusion-v1-4"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u text_to_image/save_multiple_intermediate_generation_img.py ${configuration} ${data_file} ${save_data} ${model}
