#! /bin/sh

MEMBER_FILE_PATH="--member_file_path=mia_data/cifar10_40000_whitebox_graybox_890ktrainingsteps/member_data/40000trainingsamples_member_data.joblib"
NONMEMBER_FILE_PATH="--nonmember_file_path=mia_data/cifar10_40000_whitebox_graybox_890ktrainingsteps/nonmember_data/40000trainingsamples_nonmember_data.joblib"
TRUNCATE_FLAG="--truncate_start=0 --truncate_end=-1 --num_repeat=100 --num_samples=5000"
python -u scripts/save_false_postive_rate.py ${MEMBER_FILE_PATH} ${NONMEMBER_FILE_PATH} ${TRUNCATE_FLAG}
