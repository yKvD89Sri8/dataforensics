"""
This is to extract the loss for each step
"""
import numpy as np
import pandas as pd
import os
import sys
import torch as th
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
o_path = os.getcwd()
sys.path.append(o_path)

file_path = "tmp/target_train_v2.joblib"
extract_file_path = "tmp/"

img_step_loss_dict = {} 
count = 1

with open(file_path, "rb") as fp:
    while True:
        try:
            data_list = joblib.load(fp)
            for one_point in data_list:
                imgs = one_point[0].numpy()
                losses = one_point[1]

                for i in range(imgs.shape[0]):
                    imgs_i_str = np.array2string(imgs[i], precision=4, separator=',',
                        suppress_small=True)
                    if imgs_i_str in img_step_loss_dict:
                        img_step_loss_dict[imgs_i_str].append(losses[i])
                    else:
                        img_step_loss_dict[imgs_i_str] = [losses[i]]
            #print("finish one joblib load, with data_list.size={}".format(len(data_list)))
            if len(img_step_loss_dict) == count:
                print("start to collect {}-th images data".format(count))
                count += 1
        except ValueError:
            print("all {} data points".format(len(img_step_loss_dict)))
            print("reach the end")
