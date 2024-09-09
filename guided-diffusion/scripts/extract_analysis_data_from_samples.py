import argparse
import numpy as np
import pandas as pd
import os
import sys
import torch as th
import joblib
o_path = os.getcwd()
print(o_path)
sys.path.append(o_path)
from guided_diffusion.script_util import add_dict_to_argparser

# this is the dictionary with img: loss

def extract_loss_and_log_prob(img_step_loss_dict, img_step_log_prob_dict):
    img_loss_list = []
    img_log_prob_list = []

    for one_img in img_step_loss_dict:
        img_step_loss_dict[one_img].sort()
        loss_list = [val for key, val in img_step_loss_dict[one_img]]
        img_loss_list.append(loss_list)

        img_step_log_prob_dict[one_img].sort()
        log_prob_list = [val for key, val in img_step_log_prob_dict[one_img]]
        img_log_prob_list.append(log_prob_list)
    
    return img_loss_list, img_log_prob_list

"""
extract the loss and mse data from the files
"""
def load_data(file_path, num_extract_images=50):

    img_step_loss_dict = {}
    img_step_log_prob_dict = {}

    with open(file_path, "rb") as fp:
        while True:
            try:
                data_list = joblib.load(fp)
                for one_point in data_list:
                    imgs = one_point[0].numpy()
                    losses = one_point[1]["step_losses"]
                    log_probs = one_point[1]["step_images_log_prob"]
                    t = one_point[2]

                    for i in range(imgs.shape[0]):
                        imgs_i_str = np.array2string(imgs[i], precision=4, separator=',',
                            suppress_small=True)

                        if imgs_i_str in img_step_loss_dict:
                            img_step_loss_dict[imgs_i_str].append((t[i], losses[i]))
                            img_step_log_prob_dict[imgs_i_str].append((t[i], log_probs[i]))
                        else:
                            img_step_loss_dict[imgs_i_str] = [(t[i], losses[i])]
                            img_step_log_prob_dict[imgs_i_str] = [(t[i], log_probs[i])]
                if len(img_step_log_prob_dict.keys()) == num_extract_images:
                    print("finish the loading {} images' analysis data from {}".format(len(img_step_log_prob_dict.keys()), file_path))
                    break
            except Exception as e:
                print("finish the loading data from {}".format(file_path))
                break
    return img_step_loss_dict, img_step_log_prob_dict

#save_file_path = "../tmp/temporary_data.joblib"

def main():
    args = create_argparser().parse_args()

    save_file_path = args.save_file_path
    member_file_path = args.member_file_path
    nonmember_file_path = args.nonmember_file_path
    num_extract_images = int(args.num_extract_images)

    member_img_step_loss_dict, member_img_step_log_prob_dict = load_data(member_file_path, num_extract_images)
    nonmember_img_step_loss_dict, nonmember_img_step_log_prob_dict = load_data(nonmember_file_path, num_extract_images)

    with open(save_file_path, 'ab+') as fw:
        joblib.dump({"member_img_step_loss_dict": member_img_step_loss_dict, 
                    "member_img_step_log_prob_dict": member_img_step_log_prob_dict, 
                    "nonmember_img_step_loss_dict": nonmember_img_step_loss_dict, 
                    "nonmember_img_step_log_prob_dict": nonmember_img_step_log_prob_dict}, fw)
    print("finish extracting the necessary data from {} and {} to {}".format(member_file_path, nonmember_file_path, save_file_path))

def create_argparser():
    defaults = dict(
        save_file_path = "tmp/temporary_data.joblib",
        member_file_path = "tmp/member_data.joblib",
        nonmember_file_path = "tmp/nonmember_data.joblib",
        num_extract_images = 10,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
