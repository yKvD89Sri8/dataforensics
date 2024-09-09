import joblib
import numpy as np
from scipy import interpolate
from sklearn import metrics
import random
from datetime import datetime


random.seed(datetime.now().timestamp())

member_data_file_path = "size_original/full_reconstruct_model_v1_4_member_data_laion_aesthetic_512relax_part_1040_rgb.joblib"
nonmember_data_file_path = "size_original/full_reconstruct_model_v1_4_nonmember_data_mscoco_data_part_0056_rgb.joblib"
num_sampling = 100
truncate_start = 250
truncate_end = -1
num_sampling_records = 100
nstep = 100

member_ds = joblib.load(member_data_file_path)
nonmember_ds = joblib.load(nonmember_data_file_path)

print("num_member={},num_nonmember={}".format(len(member_ds), len(nonmember_ds)) )
avg_auc_list = []
median_auc_list = []
max_auc_list = []
min_auc_list =[]
nstep_auc_list = []

for i_sampling in range(num_sampling):
    member_pred_avg_probs = [] 
    member_max_probs = []
    member_min_probs = []
    member_median_probs = []
    member_nstep_probs = []

    #if i_sampling == 0:
    random.shuffle(member_ds)
    member_ds_clip = member_ds[:num_sampling_records]
    
    for data_point in member_ds_clip:
        #one_loss.sort(reverse=True)
        one_loss = data_point["loss_trajectory"]
        one_loss.sort(reverse=True)
        #if i_sampling == 0:
        #    print("---{}".format(one_loss))
        member_pred_avg_probs.append(-np.mean(one_loss[truncate_start:truncate_end]))
        member_max_probs.append(-np.max(one_loss[truncate_start:truncate_end]))
        member_min_probs.append(-np.min(one_loss[truncate_start:truncate_end]))
        member_median_probs.append(-np.median(one_loss[truncate_start:truncate_end]))
        member_nstep_probs.append(-one_loss[nstep])

    nonmember_pred_avg_probs = []
    nonmember_max_probs = []
    nonmember_min_probs = []
    nonmember_median_probs = []
    nonmember_nstep_probs = []

    random.shuffle(nonmember_ds)
    nonmember_ds_clip = nonmember_ds[:num_sampling_records]

    for data_point in nonmember_ds_clip:
        #one_loss.sort(reverse=True)    
        one_loss = data_point["loss_trajectory"]
        one_loss.sort(reverse=True)
        nonmember_pred_avg_probs.append(-np.mean(one_loss[truncate_start:truncate_end]))
        nonmember_max_probs.append(-np.max(one_loss[truncate_start:truncate_end]))
        nonmember_min_probs.append(-np.min(one_loss[truncate_start:truncate_end]))
        nonmember_median_probs.append(-np.median(one_loss[truncate_start:truncate_end]))
        nonmember_nstep_probs.append(-one_loss[nstep])

    real_labels_positive = [1 for i in range(num_sampling_records)]
    real_labels_negative = [0 for i in range(num_sampling_records)]
    real_labels = real_labels_positive + real_labels_negative
    pred_scores_avg = [x for x in member_pred_avg_probs] + [x for x in nonmember_pred_avg_probs]
    pred_scores_median = [x for x in member_median_probs] + [x for x in nonmember_median_probs]
    pred_scores_min = [x for x in member_min_probs] + [x for x in nonmember_min_probs]
    pred_scores_max = [x for x in member_max_probs] + [x for x in nonmember_max_probs]
    pred_scores_nstep = [x for x in member_nstep_probs] + [x for x in nonmember_nstep_probs]


    avg_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_avg))
    median_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_median))
    min_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_min))
    max_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_max))
    nstep_auc_list.append(metrics.roc_auc_score(real_labels, pred_scores_nstep))

    mia_fpr,mia_tpr,mia_threshold = metrics.roc_curve(real_labels,pred_scores_avg)
    mia_avg_roc_func = interpolate.interp1d(mia_fpr, mia_tpr)
    mia_avg_tpr_at_fpr_set_dist = {"1e-1":mia_avg_roc_func(1e-1), 
                                   "1e-2":mia_avg_roc_func(1e-2), 
                                   "1e-3":mia_avg_roc_func(1e-3), 
                                   "1e-4":mia_avg_roc_func(1e-4), 
                                   "1e-5":mia_avg_roc_func(1e-5),
                                   "1e-6":mia_avg_roc_func(1e-6)}
    print("statistic: avg, auc = {}, tpr value = {}".format(avg_auc_list[-1],mia_avg_tpr_at_fpr_set_dist))

    mia_fpr,mia_tpr,mia_threshold = metrics.roc_curve(real_labels,pred_scores_median)
    mia_median_roc_func = interpolate.interp1d(mia_fpr, mia_tpr)
    mia_median_tpr_at_fpr_set_dist = {"1e-1":mia_median_roc_func(1e-1), 
                                   "1e-2":mia_median_roc_func(1e-2), 
                                   "1e-3":mia_median_roc_func(1e-3), 
                                   "1e-4":mia_median_roc_func(1e-4), 
                                   "1e-5":mia_median_roc_func(1e-5),
                                   "1e-6":mia_median_roc_func(1e-6)}
    print("statistic: median, auc = {}, tpr value = {}".format(median_auc_list[-1],mia_median_tpr_at_fpr_set_dist))

    mia_fpr,mia_tpr,mia_threshold = metrics.roc_curve(real_labels,pred_scores_min)
    mia_min_roc_func = interpolate.interp1d(mia_fpr, mia_tpr)
    mia_min_tpr_at_fpr_set_dist = {"1e-1":mia_min_roc_func(1e-1), 
                                   "1e-2":mia_min_roc_func(1e-2), 
                                   "1e-3":mia_min_roc_func(1e-3), 
                                   "1e-4":mia_min_roc_func(1e-4), 
                                   "1e-5":mia_min_roc_func(1e-5),
                                   "1e-6":mia_min_roc_func(1e-6)}
    print("statistic: min, auc = {}, value = {}".format(min_auc_list[-1], mia_min_tpr_at_fpr_set_dist))

    mia_fpr,mia_tpr,mia_threshold = metrics.roc_curve(real_labels,pred_scores_max)
    mia_max_roc_func = interpolate.interp1d(mia_fpr, mia_tpr)
    mia_max_tpr_at_fpr_set_dist = {"1e-1":mia_max_roc_func(1e-1), 
                                   "1e-2":mia_max_roc_func(1e-2), 
                                   "1e-3":mia_max_roc_func(1e-3), 
                                   "1e-4":mia_max_roc_func(1e-4), 
                                   "1e-5":mia_max_roc_func(1e-5),
                                   "1e-6":mia_max_roc_func(1e-6)}
    print("statistic: max, auc = {}, value = {}".format(max_auc_list[-1],mia_max_tpr_at_fpr_set_dist))
    
    mia_fpr,mia_tpr,mia_threshold = metrics.roc_curve(real_labels,pred_scores_nstep)
    mia_nstep_roc_func = interpolate.interp1d(mia_fpr, mia_tpr)
    mia_nstep_tpr_at_fpr_set_dist = {"1e-1":mia_nstep_roc_func(1e-1), 
                                   "1e-2":mia_nstep_roc_func(1e-2), 
                                   "1e-3":mia_nstep_roc_func(1e-3), 
                                   "1e-4":mia_nstep_roc_func(1e-4), 
                                   "1e-5":mia_nstep_roc_func(1e-5),
                                   "1e-6":mia_nstep_roc_func(1e-6)}
    print("statistic: nstep, auc = {}, tpr value = {}".format(nstep_auc_list[-1],mia_nstep_tpr_at_fpr_set_dist))

print("####################truncate_start = {}, trucnate_end = {}############".format(truncate_start, truncate_end))
print("statistic: avg, mean = {}, median = {}, min = {}, max = {}".format(
    np.mean(avg_auc_list), 
    np.median(avg_auc_list), 
    np.min(avg_auc_list), 
    np.max(avg_auc_list)))
print("statistic: median, mean = {}, median = {}, min = {}, max = {}".format(
    np.mean(median_auc_list), 
    np.median(median_auc_list), 
    np.min(median_auc_list), 
    np.max(median_auc_list)))
print("statistic: min, mean = {}, median = {}, min = {}, max = {}".format(
    np.mean(min_auc_list), 
    np.median(min_auc_list), 
    np.min(min_auc_list), 
    np.max(min_auc_list)))
print("statistic: max, mean = {}, median = {}, min = {}, max = {}".format(
    np.mean(max_auc_list), 
    np.median(max_auc_list), 
    np.min(max_auc_list), 
    np.max(max_auc_list)))
print("statistic: nstep = {}, mean = {}, median = {}, min = {}, max = {}".format(
    nstep,
    np.mean(nstep_auc_list),
    np.median(nstep_auc_list),
    np.min(nstep_auc_list),
    np.max(nstep_auc_list)))
print("#########################################################")


