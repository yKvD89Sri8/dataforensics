#! /bin/sh 

#--show_progressbar
cd ..
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --dataset_name="lambdalabs/pokemon-blip-captions" \
  --resolution=512 --center_crop --random_flip  \
  --train_batch_size=6 \
  --num_train_epochs=100 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="output" \
  --validation_prompt="a drawing of a pink rabbit"