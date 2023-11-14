'''
This script runs Experiment 3 as defined in the accompanying manuscipt. 
'''
# to run: python3 run_exp_1.py ds_lo sensor1 sensor2 aug_1 exp_name datasets

from get_data import get_data
from make_training_sets import create_LOO_train_test
import os 
from process_data import clean_all_data
from sys import argv
from training_eval import * 

# set global vars 
f = 80 # data is collected at a frequence of 80Hz
t = 5 # each activity is segmented into 5s increments 
d = 3 # for triaxial accelerometer data x, y, z

# aquire global vars from passed in args 
ds_lo = int(argv[1])
sensors = [argv[2], argv[3]]
first_training_aug = argv[4] # na, rot, flip, rf
experiment_num = argv[5]
datasets = list(argv[6:])
datasets = [int(x) for x in datasets]

print("ds_lo:", ds_lo)
print("sensors:", sensors)
print("training_aug:", first_training_aug)
print("exp num:", experiment_num)
print("datasets:", datasets)


# activity types we are considering for these experiments
labelset = ['Treadmill 2mph Lab', 'Treadmill 5.5mph Lab', 'Stationary Biking 300 Lab',
             'Walking_Up_Stairs', 'Walking_Down_Stairs', 'Sitting_Still', 'Standing_Still', 
             'Lying On Back Lab', 'Lying on Right Side Lab', 'Lying on Left Side Lab', 
             'Lying on Stomach Lab']

print(f"***** {datasets} getting data.\n")
accel, accel_start, labels, label_start = get_data(datasets, sensors, f)

print(f"***** {datasets} cleaning data.\n")
clean_accel, clean_label = clean_all_data(
    accel,
    labels, 
    accel_start, 
    label_start, 
    datasets, 
    t, 
    f, 
    d, 
    len(sensors), 
    labelset)

print(f"***** {datasets} making LOO sets.\n")

training_accel_sets, \
    training_label_sets, \
    testing_accel_sets, \
    testing_label_sets = create_LOO_train_test(
        clean_accel, 
        clean_label, 
        datasets, 
        t, 
        f, 
        d, 
        len(sensors))

for ds in datasets:
    print("\n***** breakdown:", ds)
    u, c = np.unique(clean_label[ds], return_counts=True)
    print(np.asarray((u, c)).T)

loss_path = f'RESULTS_Loss/'
acc_path = f'RESULTS_Accuracies/'
f1_path = f'RESULTS_f1s/'
conf_path = f'CONFUSION_MATRICES/'

try:
    os.mkdir(loss_path)
except FileExistsError:
    pass
try:
    os.mkdir(acc_path)
except FileExistsError:
    pass
try:
    os.mkdir(f1_path)
except FileExistsError:
    pass
try:
    os.mkdir(conf_path)
except FileExistsError:
    pass

loss_path += f'Exp_{experiment_num}/'

try:
    os.mkdir(loss_path)
except FileExistsError:
    pass

loss_path += f'{datasets}'

try:
    os.mkdir(loss_path)
except FileExistsError:
    pass

acc_path += f'Exp_{experiment_num}/' 
f1_path += f'Exp_{experiment_num}/' 
conf_path += f'Exp_{experiment_num}/'

try:
    os.mkdir(acc_path)
except FileExistsError:
    pass

try:
    os.mkdir(f1_path)
except FileExistsError:
    pass

try:
    os.mkdir(conf_path)
except FileExistsError:
    pass

acc_path += f'{datasets}'
f1_path += f'{datasets}' 
conf_path += f'{datasets}'

try:
    os.mkdir(acc_path)
except FileExistsError:
    pass

try:
    os.mkdir(f1_path)
except FileExistsError:
    pass

try:
    os.mkdir(conf_path)
except FileExistsError:
    pass

acc_path += f"/DS_{ds_lo}.csv"
f1_path += f"/DS_{ds_lo}.csv"
conf_path += f"/DS_"

print(f"***** {datasets} training.\n")

tic = time.perf_counter()
acc, f1s, conf = exp_3_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    labelset,
    ds_lo, 
    loss_path,
    first_training_aug)

make_acc_csv(acc, acc_path, experiment_num)
make_f1_csv(f1s, labelset, f1_path)
make_conf_matrix(conf,conf_path)

toc = time.perf_counter()

print(f'***** {datasets} total training took: {toc-tic} seconds.\n')