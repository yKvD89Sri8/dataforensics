import argparse
import torch
import os
import sys
import requests
import io
import json
import webdataset as wds
import numpy as np

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append("../../guided-diffusion")
print(sys.path)
#sys.path.append("{}/diffusers/examples".format(o_path))
print("{}".format(o_path))
from custom_diffusion.customized_diffusion import IntermediateOutDiffusionPipeline

from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from fastparquet import ParquetFile
from urllib.request import urlopen
from PIL import Image
from torchvision import transforms

import joblib



def extract_datapoints_webdataset_member(web_dataset, num_of_data, image_transform, ds_name="noset"):
    """
        {'similarity': 0.3301219940185547,
        'hash': -704393818681311555,
        'punsafe': 7.039978413558856e-07,
        'pwatermark': 0.09442026168107986,
        'aesthetic': 7.9463934898376465,
        'caption': 'Dinosaur Cake!-rice crispies for rocks along stream. oreo crumbles. cupcake or bought volcano cake w melted candies.',
        'url': 'http://i.pinimg.com/236x/40/d3/0b/40d30b62d2d2f088f68469937a0b57ee.jpg',
        'key': '000000052',
        'status': 'success',
        'error_message': None,
        'width': 236,
        'height': 177,
        'original_width': 236,
        'original_height': 177,
        'exif': '{"Image DateTime": "2010:03:31 12:44:00"}',
        'sha256': '71bdb3e8f3aaf8af28c9e3815c2edf0cade2ea9dcb94dd455bfa96b3b130b1fb'}
    """

    data_points_list = []
    
    for i, data_point in enumerate(web_dataset):
        try:
            if len(data_points_list) == num_of_data:
                break
            one_data_point = {}
            
            attributes = json.loads(data_point["json"].decode())
            image = Image.open(io.BytesIO(data_point["jpg"]))
            
            url = attributes["url"]
            text = attributes["caption"]
            height = attributes["original_height"]
            width = attributes["original_width"]
            
            if ds_name=="aesthetics":
                watermark_score = attributes["pwatermark"]
                aesthetic_score = attributes["aesthetic"]
                
                # if the condition doesnot satisfy, the data point is not a member
                if height <= 512 or width <= 512 or aesthetic_score <=5. or watermark_score >=0.5:
                    continue
            
            image_rgba = image_transform(image.convert("RGBA"))
            image_rgb = image_transform(image.convert("RGB"))
            image_gray = image_transform(image.convert("L"))
            
            one_data_point["text"] = text
            one_data_point["img_rgba"] = image_rgba 
            one_data_point["img_rgb"] = image_rgb
            one_data_point["img_gray"] = image_gray
            one_data_point["attributes"] = attributes
            
            data_points_list.append(one_data_point)
            logger.log("add one record, i = {}".format(i))
        except Exception as error_info:
            logger.log("error information {}({}), url: {}".format(error_info, i, url))
    
    return data_points_list

###################

def create_argparser():
    defaults = dict(
        task_type="member",
        resolution=64,
        num_generated_samples=1000, 
        save_frequency=1,
        batch_size=2, 
        data_file_path="../",
        data_save_file_path="./debug_model_v1_3_member_debug_laion2b_en.joblib",
        model_name="CompVis/stable-diffusion-v1-4",
        image_type="rgba",
        guidance_scale=7.5,
        noverbose=False,
        ds_name="default"
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
args = create_argparser().parse_args()

logger.configure(log_suffix=args.data_file_path.split("/")[-1].split(".")[0],log_prefix="save_intermediate_result")

print("args = {}".format(vars(args)))
logger.log("start process with parameters = {}".format(vars(args)))

resolution = args.resolution #64
num_of_data = args.num_generated_samples #200
save_frequency = args.save_frequency #2
data_file_path = args.data_file_path# 
data_save_file_path = args.data_save_file_path #"./member_data_with_attr.joblib"
batch_size = args.batch_size
device = "cpu" #"cuda"
guidance_scale = args.guidance_scale
image_type = args.image_type
task_type = args.task_type
model_name = args.model_name #"CompVis/stable-diffusion-v1-1"


data_file_path = args.data_file_path #"size_original/full_reconstruct_model_v1_4_member_data_laion_aesthetic_512relax_part_1040_rgb.joblib"

truncate_start = 250
truncate_end = -1
nstep = 100

data_ds = joblib.load(data_file_path)

median_probs = []

for data_point in data_ds:
    #one_loss.sort(reverse=True)
    one_loss = data_point["loss_trajectory"]
    attr = data_point["attributes"]
    one_loss.sort(reverse=True)
    
    pred_score = -np.median(one_loss[truncate_start:truncate_end])
    
    median_probs.append((pred_score, attr))

logger.log("finish data extraction with len={}".format(len(median_probs)))

if task_type=="member":
    median_probs.sort()
elif task_type =="nonmember":
    median_probs.sort(reverse=True)

pipe = IntermediateOutDiffusionPipeline.from_pretrained(model_name)

pipe.to(device)

pipe.set_progress_bar_config(disable=args.noverbose)

for i in range(30):
    caption = median_probs[i][1]["caption"]
    url = median_probs[i][1]["url"]
    logger.log("i={}, url = {}, caption = {}".format(i, url, caption))

    res = pipe(data_save_file_path, caption, guidance_scale=guidance_scale, num_inference_steps=999)
    logger.log("finish {}".format(i))
logger.log("finish")
