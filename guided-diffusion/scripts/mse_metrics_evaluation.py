import argparse
import numpy as np
import pandas as pd
import os
import sys
import torch as th
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import random
o_path = os.getcwd()
sys.path.append(o_path)
from guided_diffusion.script_util import (NUM_CLASSES, add_dict_to_argparser,
                                          args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)

# this is the dictionary with img: loss

def extract_loss_and_log_prob(img_step_loss_dict, img_step_log_prob_dict, sort_loss=True):
    img_loss_list = []
    img_log_prob_list = []

    for one_img in img_step_loss_dict:
        if sort_loss:
            img_step_loss_dict[one_img].sort() 
        loss_list = [val for key, val in img_step_loss_dict[one_img]]
        img_loss_list.append(loss_list)

        if sort_loss:
            img_step_log_prob_dict[one_img].sort()
        log_prob_list = [val for key, val in img_step_log_prob_dict[one_img]]
        img_log_prob_list.append(log_prob_list)
    
    return img_loss_list, img_log_prob_list

def extract_comprehensive_metrics(img_step_vb_dict, img_step_xstart_mse_dict, img_step_mse_dict, img_step_total_bpd_dict, img_step_prior_bpd_dict, sort_loss=True):
    img_step_vb_list = []
    img_step_xstart_mse_list= []
    img_step_mse_list = []

    img_step_total_bpd_list = []
    img_step_prior_bpd_list = []

    for one_img in img_step_vb_dict:
        if sort_loss:
            img_step_vb_dict[one_img].sort()
        step_vb_list = [val for key, val in img_step_vb_dict[one_img]]
        img_step_vb_list.append(step_vb_list)

        if sort_loss:
            img_step_xstart_mse_dict[one_img].sort()
        step_xstart_mse_list = [val for key, val in img_step_xstart_mse_dict[one_img]]
        img_step_xstart_mse_list.append(step_xstart_mse_list)

        if sort_loss:
            img_step_mse_dict[one_img].sort()
        step_mse_list = [val for key, val in img_step_mse_dict[one_img]]
        img_step_mse_list.append(step_mse_list)

        step_total_bpd_list = img_step_total_bpd_dict[one_img]
        img_step_total_bpd_list.append(step_total_bpd_list)

        step_prior_bpd_list = img_step_prior_bpd_dict[one_img]
        img_step_prior_bpd_list.append(step_prior_bpd_list)
    
    return img_step_vb_list, img_step_xstart_mse_list, img_step_mse_list, img_step_total_bpd_list, img_step_prior_bpd_list



"""
extract the loss and mse data from the files
"""
def load_data(file_path):

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
            except Exception as e:
                print("reach the end, error info={}".format(e))
                break
    return img_step_loss_dict, img_step_log_prob_dict

def load_data_wo_imgs(file_path):

    img_step_loss_dict = {}
    img_step_log_prob_dict = {}

    with open(file_path, "rb") as fp:
        while True:
            try:
                data_list = joblib.load(fp)
                for one_point in data_list:
                    imgs = one_point[0]
                    losses = one_point[1]["step_losses"]
                    log_probs = one_point[1]["step_images_log_prob"]
                    t = one_point[2]

                    for i in range(len(imgs)):
                        img_idx = imgs[i]

                        if img_idx in img_step_loss_dict:
                            img_step_loss_dict[img_idx].append((t[i], losses[i]))
                            img_step_log_prob_dict[img_idx].append((t[i], log_probs[i]))
                        else:
                            img_step_loss_dict[img_idx] = [(t[i], losses[i])]
                            img_step_log_prob_dict[img_idx] = [(t[i], log_probs[i])]
            except Exception as e:
                print("reach the end, error info={}".format(e))
                break
    return img_step_loss_dict, img_step_log_prob_dict

def load_data_vlb(file_path):

    img_step_vlb_dict = {}

    with open(file_path, "rb") as fp:
        while True:
            try:
                data_list = joblib.load(fp)
                # data_list format: image_id, vlb_score, timestep
                for one_point in data_list:
                    imgs = one_point[0]
                    vlb_score = one_point[1]
                    t = one_point[2]

                    for i in range(len(imgs)):
                        img_idx = imgs[i]

                        if img_idx in img_step_vlb_dict:
                            img_step_vlb_dict[img_idx].append((t[i], vlb_score[i]["vlb_score"]))
                        else:
                            img_step_vlb_dict[img_idx] = [(t[i], vlb_score[i]["vlb_score"])]
            except Exception as e:
                print("reach the end, error info={}".format(e))
                break
    return img_step_vlb_dict, img_step_vlb_dict

def load_data_comprehensive_vlb(file_path):

    img_step_vb_dict = {}
    img_step_xstart_mse_dict = {}
    img_step_mse_dict = {}
    img_step_total_bpd_dict = {}
    img_step_prior_bpd_dict = {}

    with open(file_path, "rb") as fp:
        while True:
            try:
                data_list = joblib.load(fp)
                # data_list format: image_id, metrics(dict, 'vb','xstart_mse','mse'), timestep, total_bpd, prior_bpd
                for one_point in data_list:
                    imgs = one_point[0]
                    metrics = one_point[1]
                    t = one_point[2]
                    total_bpd = one_point[3]
                    prior_bpd = one_point[4]

                    # for each timestep
                    for i in range(len(imgs)):
                        img_idx = imgs[i]

                        if img_idx in img_step_vb_dict:
                            img_step_vb_dict[img_idx].append((t[i], metrics[i]["vb"]))
                            img_step_xstart_mse_dict[img_idx].append((t[i], metrics[i]['xstart_mse']))
                            img_step_mse_dict[img_idx].append((t[i], metrics[i]['mse']))
                        else:
                            img_step_vb_dict[img_idx] = [(t[i], metrics[i]["vb"])]
                            img_step_xstart_mse_dict[img_idx] = [(t[i], metrics[i]['xstart_mse'])]
                            img_step_mse_dict[img_idx] = [(t[i], metrics[i]['mse'])]
                    
                    img_step_total_bpd_dict[img_idx] = total_bpd
                    img_step_prior_bpd_dict[img_idx] = prior_bpd

            except Exception as e:
                print("reach the end, error info={}".format(e))
                break
    return img_step_vb_dict, img_step_xstart_mse_dict, img_step_mse_dict, img_step_total_bpd_dict, img_step_prior_bpd_dict

# one dimension, pos_dist, neg_dist
# draw histrogram based density figure
def plot_hist(pos_dist, neg_dist, save_file):
    plt.figure()
    plt.hist(pos_dist, bins=100, alpha=0.5, weights=np.zeros_like(pos_dist) + 1. / pos_dist.size, label='positive')
    plt.hist(neg_dist, bins=100, alpha=0.5, weights=np.zeros_like(neg_dist) + 1. / neg_dist.size, label='negative')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('distance')
    plt.ylabel('normalized frequency')
    plt.savefig(save_file)
    plt.close()

#draw auc curve and calculate the number
def plot_roc(pos_results, neg_results):
    labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    return fpr, tpr, threshold, auc, ap

def compute_qualified_loss(loss_list, min_loss, max_loss, end_step, start_step=0):

    qualified_loss_list = []

    for i, one_loss in enumerate(loss_list):

        if i >= end_step and i < (start_step-1):
            continue
        if one_loss >= min_loss and one_loss <= max_loss:
            qualified_loss_list.append(one_loss)
    return qualified_loss_list

def metrics_eval_list_mode(member_data_list, nonmember_data_list, truncate_start=1000, truncate_end=-1):

    pred_avg_probs = []
    pred_medium_probs = []
    pred_max_probs = []
    pred_min_probs = []
    pred_sum_probs = []
    real_labels = []
    pred_avg_probs_and_labels = []
    pred_medium_probs_and_labels = []
    pred_max_probs_and_labels = []
    pred_min_probs_and_labels = []
    pred_sum_probs_and_labels = []

    min_loss_val = 0.01
    max_loss_val = 0.1
    last_timestep = 4000
    metrics_auc_dict = {}
    metrics_acc_dict = {}
    metrics_f1_dict = {}

    for i, member_pred_score_list in enumerate(member_data_list):

        #qualified_loss_list = compute_qualified_loss(member_pred_score_list, min_loss_val, max_loss_val, last_timestep)
        member_pred_score_list.sort(reverse=True)
        pred_avg_probs.append(-np.mean(member_pred_score_list[truncate_start:truncate_end]))
        pred_medium_probs.append(-np.median(member_pred_score_list[truncate_start:truncate_end]))
        pred_max_probs.append(-np.max(member_pred_score_list[truncate_start:truncate_end]))
        pred_min_probs.append(-np.min(member_pred_score_list[truncate_start:truncate_end]))
        pred_sum_probs.append(-np.sum(member_pred_score_list[truncate_start:truncate_end]))

        real_labels.append(1)
        
        pred_avg_probs_and_labels.append((pred_avg_probs[-1], real_labels[-1]))
        pred_medium_probs_and_labels.append((pred_medium_probs[-1], real_labels[-1]))
        pred_max_probs_and_labels.append((pred_max_probs[-1], real_labels[-1]))
        pred_min_probs_and_labels.append((pred_min_probs[-1], real_labels[-1]))
        pred_sum_probs_and_labels.append((pred_sum_probs[-1], real_labels[-1]))

    for i, nonmember_pred_score_list in enumerate(nonmember_data_list):

        #qualified_loss_list = compute_qualified_loss(nonmember_pred_score_list, min_loss_val, max_loss_val, last_timestep)
        nonmember_pred_score_list.sort(reverse=True)
        pred_avg_probs.append(-np.mean(nonmember_pred_score_list[truncate_start:truncate_end]))
        pred_medium_probs.append(-np.median(nonmember_pred_score_list[truncate_start:truncate_end]))
        pred_max_probs.append(-np.max(nonmember_pred_score_list[truncate_start:truncate_end]))
        pred_min_probs.append(-np.min(nonmember_pred_score_list[truncate_start:truncate_end]))
        pred_sum_probs.append(-np.sum(nonmember_pred_score_list[truncate_start:truncate_end]))

        real_labels.append(0)

        pred_avg_probs_and_labels.append((pred_avg_probs[-1], real_labels[-1]))
        pred_medium_probs_and_labels.append((pred_medium_probs[-1], real_labels[-1]))
        pred_max_probs_and_labels.append((pred_max_probs[-1], real_labels[-1]))
        pred_min_probs_and_labels.append((pred_min_probs[-1], real_labels[-1]))
        pred_sum_probs_and_labels.append((pred_sum_probs[-1], real_labels[-1]))

    """mia_avg_auc_score = metrics.roc_auc_score(real_labels, pred_avg_probs)
    mia_medium_auc_score = metrics.roc_auc_score(real_labels, pred_medium_probs)
    mia_max_auc_score = metrics.roc_auc_score(real_labels, pred_max_probs)
    mia_min_auc_score = metrics.roc_auc_score(real_labels, pred_min_probs)
    mia_sum_auc_score = metrics.roc_auc_score(real_labels, pred_sum_probs)

    print("mia_avg_auc = {}, mia_medium_auc = {}, mia_max_auc = {}, mia_min_auc = {}, mia_sum_auc = {}".format(mia_avg_auc_score, mia_medium_auc_score, mia_max_auc_score, mia_min_auc_score, mia_sum_auc_score))"""
    # for average calculation
    pred_avg_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_avg_probs_and_labels[int(len(pred_avg_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_avg_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_avg_auc_score = metrics.roc_auc_score(real_labels, pred_avg_probs)
    mia_avg_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_avg_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("avg: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_avg_auc_score, mia_avg_acc_score, mia_avg_f1_score))

    # for medium calculation
    pred_medium_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_medium_probs_and_labels[int(len(pred_medium_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_medium_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_medium_auc_score = metrics.roc_auc_score(real_labels, pred_medium_probs)
    mia_medium_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_medium_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("medium: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_medium_auc_score, mia_medium_acc_score, mia_medium_f1_score))

    # for max calculation
    pred_max_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_max_probs_and_labels[int(len(pred_max_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_max_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_max_auc_score = metrics.roc_auc_score(real_labels, pred_max_probs)
    mia_max_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_max_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("max: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_max_auc_score, mia_max_acc_score, mia_max_f1_score))

    # for min calculation
    pred_min_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_min_probs_and_labels[int(len(pred_min_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_min_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_min_auc_score = metrics.roc_auc_score(real_labels, pred_min_probs)
    mia_min_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_min_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("min: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_min_auc_score, mia_min_acc_score, mia_min_f1_score))

    # for sum calculation
    pred_sum_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_sum_probs_and_labels[int(len(pred_sum_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_sum_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_sum_auc_score = metrics.roc_auc_score(real_labels, pred_sum_probs)
    mia_sum_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_sum_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("sum: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_sum_auc_score, mia_sum_acc_score, mia_sum_f1_score))

    metrics_auc_dict['min'] = mia_min_auc_score
    metrics_auc_dict['max'] = mia_max_auc_score
    metrics_auc_dict['medium'] = mia_medium_auc_score
    metrics_auc_dict['avg'] = mia_avg_auc_score
    metrics_auc_dict['sum'] = mia_sum_auc_score

    metrics_acc_dict['min'] = mia_min_acc_score
    metrics_acc_dict['max'] = mia_max_acc_score
    metrics_acc_dict['medium'] = mia_medium_acc_score
    metrics_acc_dict['avg'] = mia_avg_acc_score
    metrics_acc_dict['sum'] = mia_sum_acc_score

    metrics_f1_dict['min'] = mia_min_f1_score
    metrics_f1_dict['max'] = mia_max_f1_score
    metrics_f1_dict['medium'] = mia_medium_f1_score
    metrics_f1_dict['avg'] = mia_avg_f1_score
    metrics_f1_dict['sum'] = mia_sum_f1_score

    return metrics_auc_dict, metrics_acc_dict, metrics_f1_dict

def metrics_eval_list_and_loss_filter_mode(member_data_list, nonmember_data_list):

    pred_avg_probs = []
    pred_medium_probs = []
    pred_max_probs = []
    pred_min_probs = []
    pred_sum_probs = []
    real_labels = []
    pred_avg_probs_and_labels = []
    pred_medium_probs_and_labels = []
    pred_max_probs_and_labels = []
    pred_min_probs_and_labels = []
    pred_sum_probs_and_labels = []

    min_loss_val = 0.01
    max_loss_val = 0.1
    last_timestep = 4000
    metrics_auc_dict = {}
    metrics_acc_dict = {}
    metrics_f1_dict = {}

    for i, member_pred_score_list in enumerate(member_data_list):

        qualified_loss_list = compute_qualified_loss(member_pred_score_list, min_loss_val, max_loss_val, last_timestep)
        #member_pred_score_list.sort(reverse=True)
        pred_avg_probs.append(-np.mean(qualified_loss_list))
        pred_medium_probs.append(-np.median(qualified_loss_list))
        pred_max_probs.append(-np.max(qualified_loss_list))
        pred_min_probs.append(-np.min(qualified_loss_list))
        pred_sum_probs.append(-np.sum(qualified_loss_list))

        real_labels.append(1)
        
        pred_avg_probs_and_labels.append((pred_avg_probs[-1], real_labels[-1]))
        pred_medium_probs_and_labels.append((pred_medium_probs[-1], real_labels[-1]))
        pred_max_probs_and_labels.append((pred_max_probs[-1], real_labels[-1]))
        pred_min_probs_and_labels.append((pred_min_probs[-1], real_labels[-1]))
        pred_sum_probs_and_labels.append((pred_sum_probs[-1], real_labels[-1]))

    for i, nonmember_pred_score_list in enumerate(nonmember_data_list):

        qualified_loss_list = compute_qualified_loss(nonmember_pred_score_list, min_loss_val, max_loss_val, last_timestep)
        #nonmember_pred_score_list.sort(reverse=True)
        pred_avg_probs.append(-np.mean(qualified_loss_list))
        pred_medium_probs.append(-np.median(qualified_loss_list))
        pred_max_probs.append(-np.max(qualified_loss_list))
        pred_min_probs.append(-np.min(qualified_loss_list))
        pred_sum_probs.append(-np.sum(qualified_loss_list))

        real_labels.append(0)

        pred_avg_probs_and_labels.append((pred_avg_probs[-1], real_labels[-1]))
        pred_medium_probs_and_labels.append((pred_medium_probs[-1], real_labels[-1]))
        pred_max_probs_and_labels.append((pred_max_probs[-1], real_labels[-1]))
        pred_min_probs_and_labels.append((pred_min_probs[-1], real_labels[-1]))
        pred_sum_probs_and_labels.append((pred_sum_probs[-1], real_labels[-1]))

    """mia_avg_auc_score = metrics.roc_auc_score(real_labels, pred_avg_probs)
    mia_medium_auc_score = metrics.roc_auc_score(real_labels, pred_medium_probs)
    mia_max_auc_score = metrics.roc_auc_score(real_labels, pred_max_probs)
    mia_min_auc_score = metrics.roc_auc_score(real_labels, pred_min_probs)
    mia_sum_auc_score = metrics.roc_auc_score(real_labels, pred_sum_probs)

    print("mia_avg_auc = {}, mia_medium_auc = {}, mia_max_auc = {}, mia_min_auc = {}, mia_sum_auc = {}".format(mia_avg_auc_score, mia_medium_auc_score, mia_max_auc_score, mia_min_auc_score, mia_sum_auc_score))"""
    # for average calculation
    pred_avg_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_avg_probs_and_labels[int(len(pred_avg_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_avg_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_avg_auc_score = metrics.roc_auc_score(real_labels, pred_avg_probs)
    mia_avg_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_avg_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("avg: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_avg_auc_score, mia_avg_acc_score, mia_avg_f1_score))

    # for medium calculation
    pred_medium_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_medium_probs_and_labels[int(len(pred_medium_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_medium_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_medium_auc_score = metrics.roc_auc_score(real_labels, pred_medium_probs)
    mia_medium_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_medium_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("medium: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_medium_auc_score, mia_medium_acc_score, mia_medium_f1_score))

    # for max calculation
    pred_max_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_max_probs_and_labels[int(len(pred_max_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_max_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_max_auc_score = metrics.roc_auc_score(real_labels, pred_max_probs)
    mia_max_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_max_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("max: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_max_auc_score, mia_max_acc_score, mia_max_f1_score))

    # for min calculation
    pred_min_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_min_probs_and_labels[int(len(pred_min_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_min_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_min_auc_score = metrics.roc_auc_score(real_labels, pred_min_probs)
    mia_min_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_min_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("min: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_min_auc_score, mia_min_acc_score, mia_min_f1_score))

    # for sum calculation
    pred_sum_probs_and_labels.sort()
    pred_labels_and_real_labels = [(1,x[1]) if x[0] > pred_sum_probs_and_labels[int(len(pred_sum_probs_and_labels)/2)][0] else (0,x[1]) for x in pred_sum_probs_and_labels]
    y_pred = [x[0] for x in pred_labels_and_real_labels]
    y_labels = [x[1] for x in pred_labels_and_real_labels]

    mia_sum_auc_score = metrics.roc_auc_score(real_labels, pred_sum_probs)
    mia_sum_acc_score = metrics.accuracy_score(y_labels, y_pred)
    mia_sum_f1_score = metrics.f1_score(y_labels, y_pred)
    #print("sum: auc score = {}, accuracy score = {}, f1 score = {}".format(mia_sum_auc_score, mia_sum_acc_score, mia_sum_f1_score))

    metrics_auc_dict['min'] = mia_min_auc_score
    metrics_auc_dict['max'] = mia_max_auc_score
    metrics_auc_dict['medium'] = mia_medium_auc_score
    metrics_auc_dict['avg'] = mia_avg_auc_score
    metrics_auc_dict['sum'] = mia_sum_auc_score

    metrics_acc_dict['min'] = mia_min_acc_score
    metrics_acc_dict['max'] = mia_max_acc_score
    metrics_acc_dict['medium'] = mia_medium_acc_score
    metrics_acc_dict['avg'] = mia_avg_acc_score
    metrics_acc_dict['sum'] = mia_sum_acc_score

    metrics_f1_dict['min'] = mia_min_f1_score
    metrics_f1_dict['max'] = mia_max_f1_score
    metrics_f1_dict['medium'] = mia_medium_f1_score
    metrics_f1_dict['avg'] = mia_avg_f1_score
    metrics_f1_dict['sum'] = mia_sum_f1_score

    return metrics_auc_dict, metrics_acc_dict, metrics_f1_dict


def balance_data(member_data_list, nonmember_data_list):

    if len(member_data_list) > len(nonmember_data_list):
        member_data_list = member_data_list[:len(nonmember_data_list)]
    elif len(member_data_list) < len(nonmember_data_list):
        nonmember_data_list = nonmember_data_list[:len(member_data_list)]

    print("number of member = {}; number of nonmember = {}".format(len(member_data_list), len(nonmember_data_list)))

    return member_data_list, nonmember_data_list

def merge2list(list_dict, one_dict):
    list_dict["min"].append(one_dict["min"])
    list_dict["max"].append(one_dict["max"])
    list_dict["medium"].append(one_dict["medium"])
    list_dict["avg"].append(one_dict["avg"])
    list_dict["sum"].append(one_dict["sum"])

def print_list_dict(list_dict, headline="headline"):
    print("############{}#################".format(headline))
    min_val_list = list_dict["min"]
    if len(min_val_list) >0:
        print("min | mean = {}, median = {}, max = {}, min = {}".format(np.mean(min_val_list), np.median(min_val_list), np.max(min_val_list), np.min(min_val_list)))
    
    max_val_list = list_dict["max"]
    if len(max_val_list) > 0:
        print("max | mean = {}, median = {}, max = {}, min = {}".format(np.mean(max_val_list), np.median(max_val_list), np.max(max_val_list), np.min(max_val_list)))
    
    medium_val_list = list_dict["medium"]
    if len(medium_val_list) >0:
        print("medium | mean = {}, median = {}, max = {}, min = {}".format(np.mean(medium_val_list), np.median(medium_val_list), np.max(medium_val_list), np.min(medium_val_list)))
    
    avg_val_list = list_dict["avg"]
    if len(avg_val_list) > 0:
        print("avg | mean = {}, median = {}, max = {}, min = {}".format(np.mean(avg_val_list), np.median(avg_val_list), np.max(avg_val_list), np.min(avg_val_list)))
    
    sum_val_list = list_dict["sum"]
    if len(sum_val_list) > 0:
        print("sum | mean = {}, median = {}, max = {}, min = {}".format(np.mean(sum_val_list), np.median(sum_val_list), np.max(sum_val_list), np.min(sum_val_list)))

def print_scale_values(data_list, headline="headlines"):
    print("######################{}###################".format(headline))
    print("position | mean = {}, max = {}, min = {}".format(np.mean(data_list), np.max(data_list), np.min(data_list)))

def find_pos(data_lists, range_start=0.01, range_end=0.1):
    pos_list = []
    for one_data_point in data_lists:
        for i, one_point in enumerate(one_data_point):
            if one_point >= range_start and one_point < range_end:
                pos_list.append(i)
                break
    return pos_list

def create_argparser():
    defaults = dict(member_file_path = "guided-diffusion/tmp/loss_analysis_celeba_gender_10000samples_2023-01-28_10-31-32/10000trainingsamples_member_data.joblib",
    nonmember_file_path = "guided-diffusion/tmp/loss_analysis_celeba_gender_10000samples_2023-01-28_10-31-05/10000trainingsamples_nonmember_data.joblib",
    truncate_start = 1000,
    truncate_end = -1,
    loss_filter = False,
    num_repeat = 1000,
    num_samples = 1000,
    range_start = 0.01,
    range_end = 0.1
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()

member_file_path = args.member_file_path
nonmember_file_path = args.nonmember_file_path
range_start = args.range_start
range_end = args.range_end

print("member_file_path = {}".format(member_file_path))
print("nonmember_file_path = {}".format(nonmember_file_path))

member_img_step_mse_dict, member_img_step_log_prob_dict = load_data_wo_imgs(member_file_path)
nonmember_img_step_mse_dict, nonmember_img_step_log_prob_dict = load_data_wo_imgs(nonmember_file_path)

member_img_mse_list, member_img_log_prob_list = extract_loss_and_log_prob(member_img_step_mse_dict, member_img_step_log_prob_dict, sort_loss=False)
nonmember_img_mse_list, nonmember_img_log_prob_list = extract_loss_and_log_prob(nonmember_img_step_mse_dict, nonmember_img_step_log_prob_dict, sort_loss=False)

member_img_mse_list = member_img_mse_list[:-1]
nonmember_img_mse_list = nonmember_img_mse_list[:-1]

if len(member_img_mse_list) > len(nonmember_img_mse_list):
    member_img_mse_list = member_img_mse_list[:len(nonmember_img_mse_list)]
elif len(member_img_mse_list) < len(nonmember_img_mse_list):
    nonmember_img_mse_list = nonmember_img_mse_list[:len(member_img_mse_list)]
print("number of member = {}; number of nonmember = {}".format(len(member_img_mse_list), len(nonmember_img_mse_list)))

member_pos_set = find_pos(member_img_mse_list, range_start, range_end)
nonmember_pos_set = find_pos(nonmember_img_mse_list, range_start, range_end)
print_scale_values(member_pos_set, "member data first place larger than {}".format(range_start))

xstart_mse_auc_list_dict = {"min":[],"max":[],"medium":[],"avg":[],"sum":[]}
xstart_mse_acc_list_dict = {"min":[],"max":[],"medium":[],"avg":[],"sum":[]}
xstart_mse_f1_list_dict = {"min":[],"max":[],"medium":[],"avg":[],"sum":[]}

num_repeat = args.num_repeat
num_samples = args.num_samples

print("run {} times sampling, each sampling with {} records".format(num_repeat, num_samples))

if args.loss_filter == False:
    print("mode: truncate_start = {}, truncate_end = {}".format(args.truncate_start, args.truncate_end))
    for idx in range(num_repeat):
        random.seed(idx)
        random.shuffle(member_img_mse_list)
        random.shuffle(nonmember_img_mse_list)
        xstart_mse_auc, xstart_mse_acc, xstart_mse_f1 = metrics_eval_list_mode(member_img_mse_list[:num_samples], nonmember_img_mse_list[:num_samples], truncate_start=args.truncate_start, truncate_end=args.truncate_end)
        merge2list(xstart_mse_auc_list_dict, xstart_mse_auc)
        merge2list(xstart_mse_acc_list_dict, xstart_mse_acc)
        merge2list(xstart_mse_f1_list_dict, xstart_mse_f1)

else:
    print("mode: loss filter")
    for idx in range(num_repeat):
        random.seed(idx)
        random.shuffle(member_img_mse_list)
        random.shuffle(nonmember_img_mse_list)
        xstart_mse_auc, xstart_mse_acc, xstart_mse_f1 = metrics_eval_list_and_loss_filter_mode(member_img_mse_list, nonmember_img_mse_list)
        merge2list(xstart_mse_auc_list_dict, xstart_mse_auc)
        merge2list(xstart_mse_acc_list_dict, xstart_mse_acc)
        merge2list(xstart_mse_f1_list_dict, xstart_mse_f1)

print_list_dict(xstart_mse_auc_list_dict, headline="xstart mse auc")
print_list_dict(xstart_mse_acc_list_dict, headline="xstart mse acc")
print_list_dict(xstart_mse_f1_list_dict, headline="xstart mse f1 score")
