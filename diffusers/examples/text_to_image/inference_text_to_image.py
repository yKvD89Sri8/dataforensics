import torch
import os
import sys
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL,UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_safe.safety_checker import SafeStableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from typing import Any, Callable, Dict, List, Optional, Union
o_path = os.getcwd()
sys.path.append(o_path)
#sys.path.append("{}/diffusers/examples".format(o_path))
print("{}".format(o_path))
from custom_diffusion.customized_diffusion import CustomizedDiffusionPipeline

from fastparquet import ParquetFile
from urllib.request import urlopen
from PIL import Image
from torchvision import transforms

import joblib

resolution = 64

image_transforms = transforms.Compose(
    [
        transforms.Resize(size=(resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        #transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        #transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5]),
    ]
)

####### load member data#####
num_data = 200
member_data_file = "diffusers/data/laion2B-en/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet"
member_pf = ParquetFile(member_data_file)
member_pd_data = member_pf.to_pandas().sample(n=2000,ignore_index=True)

validation_list = []

for i in range(num_data):
    try:
        one_data_point = {}
        url = member_pd_data["URL"][i]
        text = member_pd_data["TEXT"][i]
        image = Image.open(urlopen(url))
        image_rgba = image_transforms(image.convert("RGBA"))
        image_rgb = image_transforms(image.convert("RGB"))
        
        one_data_point["text"] = text
        one_data_point["img_rgb"] = image_rgb
        one_data_point["img_rgba"] = image_rgba
        
        validation_list.append(one_data_point)
    except:
        print("error in {} step".format(i))
    
    #pil_img = transforms.functional.to_pil_image(trans_img)
    #pil_img.save("test_2.png")

# runwayml/stable-diffusion-v1-5
print(os.getcwd())
device = "cuda"
model_path = "./examples/text_to_image/output/checkpoint-5000/pytorch_model.bin"
"""pipe = CustomizedDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)"""
pipe = CustomizedDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-1"
)
# load lora weights
#pipe.unet.load_attn_procs(model_path)
# set to use GPU for inference
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

# generate image
#prompt = "a photo of an astronaut riding a horse on mars"
loss_list = []
for i, data in enumerate(validation_list):
    prompt = validation_list[i]["text"]
    groundtruth_img = validation_list[i]["img_rgba"]
    #image = pipe(prompt, groundtruth_image=groundtruth_img, num_inference_steps=100).images[0]
    res = pipe(prompt, groundtruth_image=groundtruth_img, num_inference_steps=999)
    if res.nsfw_content_detected[0][0] == False:
        loss_list.append(res.nsfw_content_detected[1])
    print("finish one")
    if (i+1) % 20 == 0:
        joblib.dump(loss_list, "./member_data_200records.joblib")
        print("in {} step, save in {}".format(i+1, "./member_data_200records.joblib" ))
joblib.dump(loss_list, "./member_data_20records.joblib")
# save image
#image.save("image_final.png")
print("finish with {} records".format(len(loss_list)))