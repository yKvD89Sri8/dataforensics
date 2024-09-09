import torch
from torch import autocast 
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from fastparquet import ParquetFile


# to load a parquet file
file_path = "diffusers/data/laion2b_chinese_release/train-00000-of-00013.parquet"
pf = ParquetFile(file_path)
pd_data = pf.to_pandas()

# to load a image from online 
from urllib.request import urlopen
from PIL import Image

url="https://img.alicdn.com/imgextra/i3/817462628/O1CN01eLHBGX1VHfUMBA1du_!!817462628.jpg"
img = Image.open(urlopen(url))

#img_rgb = img.convert("RGBA")
img_rgb = img.convert("RGB")


"""
model_id = "CompVis/stable-diffusion-v1-1"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"

with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
"""

data_file_path ="diffusers/examples/nonmember_data.joblib"
nonmember_ds = joblib.load(data_file_path)

ds_member_data = {}
count=0
for loss_list in nonmember_ds:
    for i, val in enumerate(loss_list):
        if i<3:
            continue
        ds_member_data[count] = [4000-i, "nonmember_data_loss_trace", val]
        count = count+1

member_pred_avg_probs = []
member_max_probs = []
member_min_probs = []
member_median_probs = []

truncate_start = 500
truncate_end = -1
for one_loss in member_ds:
    member_pred_avg_probs.append(np.mean(one_loss[truncate_start:truncate_end]))
    member_max_probs.append(np.max(one_loss[truncate_start:truncate_end]))
    member_min_probs.append(np.min(one_loss[truncate_start:truncate_end]))
    member_median_probs.append(np.median(one_loss[truncate_start:truncate_end]))

nonmember_pred_avg_probs = []
nonmember_max_probs = []
nonmember_min_probs = []
nonmember_median_probs = []

truncate_start = 500
truncate_end = -1
for one_loss in nonmember_ds:
    nonmember_pred_avg_probs.append(np.mean(one_loss[truncate_start:truncate_end]))
    nonmember_max_probs.append(np.max(one_loss[truncate_start:truncate_end]))
    nonmember_min_probs.append(np.min(one_loss[truncate_start:truncate_end]))
    nonmember_median_probs.append(np.median(one_loss[truncate_start:truncate_end]))

real_labels_positive = [1 for i in range(len(member_ds))]
real_labels_negative = [0 for i in range(len(nonmember_ds))]
real_labels = real_labels_positive + real_labels_negative
pred_scores_avg = [x for x in member_pred_avg_probs] + [x for x in nonmember_pred_avg_probs]
pred_scores_median = [x for x in member_median_probs] + [x for x in nonmember_median_probs]
pred_scores_min = [x for x in member_min_probs] + [x for x in nonmember_min_probs]
pred_scores_max = [x for x in member_max_probs] + [x for x in nonmember_max_probs]

#######################

sample_num = 20
avg_auc_list = []
median_auc_list = []
max_auc_list = []
min_auc_list =[]
for one_sample in range(sample_num):
    member_pred_avg_probs = []
    member_max_probs = []
    member_min_probs = []
    member_median_probs = []

    truncate_start = 750
    truncate_end = -1
    for one_loss in member_ds:
        member_pred_avg_probs.append(np.mean(one_loss[truncate_start:truncate_end]))
        member_max_probs.append(np.max(one_loss[truncate_start:truncate_end]))
        member_min_probs.append(np.min(one_loss[truncate_start:truncate_end]))
        member_median_probs.append(np.median(one_loss[truncate_start:truncate_end]))

    nonmember_pred_avg_probs = []
    nonmember_max_probs = []
    nonmember_min_probs = []
    nonmember_median_probs = []

    truncate_start = 750
    truncate_end = -1
    random.shuffle(nonmember_ds)
    nonmember_ds_clip = nonmember_ds[:len(member_ds)]
    for one_loss in nonmember_ds_clip:
        nonmember_pred_avg_probs.append(np.mean(one_loss[truncate_start:truncate_end]))
        nonmember_max_probs.append(np.max(one_loss[truncate_start:truncate_end]))
        nonmember_min_probs.append(np.min(one_loss[truncate_start:truncate_end]))
        nonmember_median_probs.append(np.median(one_loss[truncate_start:truncate_end]))

    real_labels_positive = [1 for i in range(len(member_ds))]
    real_labels_negative = [0 for i in range(len(member_ds))]
    real_labels = real_labels_positive + real_labels_negative
    pred_scores_avg = [x for x in member_pred_avg_probs] + [x for x in nonmember_pred_avg_probs]
    pred_scores_median = [x for x in member_median_probs] + [x for x in nonmember_median_probs]
    pred_scores_min = [x for x in member_min_probs] + [x for x in nonmember_min_probs]
    pred_scores_max = [x for x in member_max_probs] + [x for x in nonmember_max_probs]

    avg_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_avg))
    median_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_median))
    min_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_min))
    max_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_max))

np.mean(min_auc_list),np.median(min_auc_list),np.min(min_auc_list),np.max(min_auc_list)


#### usage of webtestdataset

import webdataset as wds
from PIL import Image
import io
url = "diffusers/data/mscoco/mscoco/{00000..00003}.tar"
dataset = wds.WebDataset(url)
for sample in dataset:
    sample
    
image_data = sample["jpg"]
image = Image.open(io.BytesIO(image_data))
image.show()
image_rgb = image.convert("RGBA")
byte_value = sample["json"].decode().replace("'",'"')
d = json.loads(byte_value)