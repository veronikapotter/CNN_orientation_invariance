# This file contains functions to train and evaluate each experiment in the the accompanying 
# manuscript.
import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score
import tensorflow as tf 
import time 
from tensorflow.keras.callbacks import CSVLogger
from models import *
from make_training_sets import * 

# Make a confusion matrix. 
# Params: lab_pool (list) - activities considered across experiments, true (array) - the true labels 
# of an evaluation set, preds (arrary) - the model's predictions across an evaluation set. 
# Returns: mat_df (DataFrame) - the confusion matrix (rows = true, cols = preds).
def confusion_matrix(lab_pool, true, preds):
    mat_df = pd.DataFrame(columns=[lab_pool], index=lab_pool)

    unmapped_true = unmap_labels(true, lab_pool)
    unmapped_preds = unmap_labels(preds, lab_pool)

    for lab in lab_pool:
        mat_df[lab] = 0

    for i in range(len(unmapped_true)):
        mat_df.loc[unmapped_true[i]][unmapped_preds[i]] = \
            mat_df.loc[unmapped_true[i]][unmapped_preds[i]] + 1
    return mat_df

# Map labels from a string to a one-hot-vector.
# Params: labels (array) - labels for training/evalutation sets, lab_pool (list) - activities 
# considered across experiments.
# Returns: mapped_labels (array) - all labels as one-hot-vectors. 
def map_labels(labels, lab_pool):
    mapped_labels = []
    for label in labels: 
        map_l = np.zeros(len(lab_pool))
        i = lab_pool.index(label)
        map_l[i] = 1
        mapped_labels.append(map_l)

    return np.array(mapped_labels)

# Map labels from a one-hot-vector to a string.
# Params: labels (array) - labels for training/evalutation sets (as one-hot-vectors), lab_pool 
# (list) - activities considered across experiments.
# Returns: unmapped (array) - all labels as their original strings. 
def unmap_labels(labels, lab_pool):
    unmapped = []
    for lab in labels:
        unmapped.append(unmap_label(lab, lab_pool))
    return unmapped

# Map a label from a one-hot-vector to a string.
# Params: label (string) - a label (as a one-hot-vector), lab_pool (list) - activities considered 
# across experiments.
# Returns: unmapped (string) - the labels as its original string. 
def unmap_label(label, lab_pool):
    ind = np.argmax(label)
    if ind <= len(lab_pool):
        return lab_pool[ind]
    else:
        return "catch all"

# Create a csv containing reported accuracies across all evaluation sets and seeds. 
# Params: acc (dict) - evaluated accuracies across evaluation sets for all seeds, file (string) - 
# path to where accuracies are reported, exp (string) - the experiment number.
# Returns: nothing. 
def make_acc_csv(acc, file, exp):
    if "1" in exp:
        cols = [
            'DS Left Out', 
            'Training 1 Acc (non-aug)', 
            'Training 1 Acc (rots)',
            'Training 1 Acc (flips)',
            'Training 1 Acc (rots and flips)',
            'Training 2 Acc (non-aug)', 
            'Training 2 Acc (rots)',
            'Training 2 Acc (flips)',
            'Training 2 Acc (rots and flips)',
            'Training 3 Acc (non-aug)', 
            'Training 3 Acc (rots)',
            'Training 3 Acc (flips)',
            'Training 3 Acc (rots and flips)',
            'Training Time (all models)']
    elif "2" in exp:
        cols  = [
            'DS Left Out', 
            'Training 1 Acc (NA-NA)', 
            'Training 1 Acc (NA-R)',
            'Training 1 Acc (NA-F)',
            'Training 1 Acc (NA-RF)',
            'Training 1 Acc (R-NA)', 
            'Training 1 Acc (R-R)',
            'Training 1 Acc (R-F)',
            'Training 1 Acc (R-RF)',
            'Training 1 Acc (F-NA)', 
            'Training 1 Acc (F-R)',
            'Training 1 Acc (F-F)',
            'Training 1 Acc (F-RF)',
            'Training 1 Acc (RF-NA)', 
            'Training 1 Acc (RF-R)',
            'Training 1 Acc (RF-F)',
            'Training 1 Acc (RF-RF)',
            'Training 2 Acc (NA-NA)', 
            'Training 2 Acc (NA-R)',
            'Training 2 Acc (NA-F)',
            'Training 2 Acc (NA-RF)',
            'Training 2 Acc (R-NA)', 
            'Training 2 Acc (R-R)',
            'Training 2 Acc (R-F)',
            'Training 2 Acc (R-RF)',
            'Training 2 Acc (F-NA)', 
            'Training 2 Acc (F-R)',
            'Training 2 Acc (F-F)',
            'Training 2 Acc (F-RF)',
            'Training 2 Acc (RF-NA)', 
            'Training 2 Acc (RF-R)',
            'Training 2 Acc (RF-F)',
            'Training 2 Acc (RF-RF)', 
            'Training 3 Acc (NA-NA)', 
            'Training 3 Acc (NA-R)',
            'Training 3 Acc (NA-F)',
            'Training 3 Acc (NA-RF)',
            'Training 3 Acc (R-NA)', 
            'Training 3 Acc (R-R)',
            'Training 3 Acc (R-F)',
            'Training 3 Acc (R-RF)',
            'Training 3 Acc (F-NA)', 
            'Training 3 Acc (F-R)',
            'Training 3 Acc (F-F)',
            'Training 3 Acc (F-RF)',
            'Training 3 Acc (RF-NA)', 
            'Training 3 Acc (RF-R)',
            'Training 3 Acc (RF-F)',
            'Training 3 Acc (RF-RF)',         
            'Training Time (all models)']
    elif "3" in exp:
        cols = [
            'DS Left Out', 
            'Training 1 Acc (non-aug)', 
            'Training 1 Acc (rw)',
            'Training 1 Acc (rots)',
            'Training 1 Acc (flips)',
            'Training 1 Acc (rots and flips)',
            'Training 2 Acc (non-aug)', 
            'Training 2 Acc (rw)',
            'Training 2 Acc (rots)',
            'Training 2 Acc (flips)',
            'Training 2 Acc (rots and flips)',
            'Training 3 Acc (non-aug)', 
            'Training 3 Acc (rw)',
            'Training 3 Acc (rots)',
            'Training 3 Acc (flips)',
            'Training 3 Acc (rots and flips)',
            'Training Time (all models)']
        
    np_acc = np.array(list(acc.values()))
    df = pd.DataFrame(np_acc, columns=cols)
    df.to_csv(file)

# Create a csv containing reported f1s across all evaluation sets and seeds. 
# Params: f1s (dict) - evaluated f1s across evaluation sets for all seeds, labs (list) - 
# activities considered across experiments, file (string) - path to where f1s are reported.
# Returns: nothing.     
def make_f1_csv(f1s, labs, file):
    cols = [x for x in labs]
    cols.insert(0, 'Training #')
    cols.insert(0, 'DS Left Out')
    cols.append("F1 (avg)")
    np_f1 = np.array(list(f1s.values()))
    df = pd.DataFrame(np_f1, columns=cols)
    df.to_csv(file)

# Create a csv containing the confusion matrix across all evaluation sets and seeds. 
# Params: conf (DataFrame) - the confusion matrix for an evaluation set, file (string) - 
# path to where confusion matrices are reported.
# Returns: nothing. 
def make_conf_matrix(conf, file):
    for key in conf.keys():
        path = file + f'{key}.csv'
        conf[key].to_csv(path)

# Train and evaluate according to the methodology of Experiment 1 in the accompanying manuscript.
# Params: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data, label_set (dict) - activities considered across 
# experiments, ds_lo (int) - the ID of the left out participant, path (string) - where loss results 
# will be written, training_aug (string) - the orientations to be used in the training set.
# Returns: accuracies (dict) - evaluated accuracies across evaluation sets for all seeds, f1_scores 
# (dict) - evaluated f1 scores across evaluation sets for all seeds, confusion_matrices (dict) - 
# confusion matrices across evaluation sets for all seeds. 
def exp_1_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    label_set,
    ds_lo,
    path,
    training_aug):

    accuracies = {}
    f1_scores = {}
    confusion_matrices = {}

    # these dicts use the keys of the dataset aug (e.g., na, rots, flips, rfs).  
    eval_labels_aug = make_labels_for_aug_data(testing_label_sets[ds_lo])
    eval_accel_aug = create_augmented_datasets(testing_accel_sets[ds_lo], 0)
    train_labels_aug = make_labels_for_aug_data(training_label_sets[ds_lo])
    train_accel_aug = create_augmented_datasets(training_accel_sets[ds_lo], 0)

    if training_aug != "na":
        train_accel = np.concatenate((train_accel_aug["na"], train_accel_aug[training_aug]))
        train_labels = map_labels(
            np.concatenate((train_labels_aug["na"], train_labels_aug[training_aug])), 
            label_set)
    else:
        train_accel = train_accel_aug["na"]
        train_labels = map_labels(train_labels_aug["na"], label_set)

    accuracies[ds_lo] = [ds_lo]
    t1 = time.perf_counter()
    input_shape = train_accel.shape[1:]
    output_shape = train_labels.shape[1]

    # train and eval three random seeds with the same training set 
    for i in range(1, 4): 
            tf.keras.utils.disable_interactive_logging()
            
            # define, compile, and train model 
            cnn = single_sensor_CNN(input_shape, output_shape)
            cnn.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])
            csv_logger = CSVLogger(
                f'{path}/DS_{ds_lo}_Loss_Training_{i}.csv', 
                append=True, separator=',')
            cnn.fit(
                train_accel, 
                train_labels, 
                batch_size=100, 
                epochs=25,
                callbacks=[csv_logger])
        
            # evaluate model on all orientations of the left out ds 
            # TODO: refactor this code to be a method
            results = cnn.evaluate(eval_accel_aug["na"], map_labels(
                eval_labels_aug["na"], 
                label_set))
            acc = round(results[1]*100, 2)
            accuracies[ds_lo].append(acc)
            results = cnn.evaluate(eval_accel_aug["rots"], map_labels(
                eval_labels_aug["rots"], 
                label_set))
            acc = round(results[1]*100, 2)
            accuracies[ds_lo].append(acc)
            results = cnn.evaluate(eval_accel_aug["flips"], map_labels(
                eval_labels_aug["flips"], 
                label_set))
            acc = round(results[1]*100, 2)
            accuracies[ds_lo].append(acc)
            results = cnn.evaluate(eval_accel_aug["rfs"], map_labels(
                eval_labels_aug["rfs"], 
                label_set))
            acc = round(results[1]*100, 2)
            accuracies[ds_lo].append(acc)
                
            # generate f1 scores for each orientation of the left out ds
            # TODO: refactor this code to be a method
            preds = cnn.predict(eval_accel_aug["na"])
            y_pred = np.argmax(preds, axis=1)
            y_true = np.argmax(map_labels(eval_labels_aug["na"], label_set), axis=1)
            f1 = list(f1_score(y_true, y_pred, average=None, labels=np.arange(output_shape)))
            f1.append(np.average(f1)) # add the average to the very end 
            f1.insert(0, i) # add training num
            f1.insert(0, ds_lo) # add DS left out 
            f1_scores[f'{ds_lo} training {i}'] = f1
            
            rot_preds = cnn.predict(eval_accel_aug["rots"])
            y_rot_pred = np.argmax(rot_preds, axis=1)
            y_rot_true = np.argmax(map_labels(eval_labels_aug["rots"], label_set), axis=1)
            f1 = list(f1_score(y_rot_true, y_rot_pred, average=None, labels=np.arange(output_shape)))
            f1.append(np.average(f1)) # add the average to the very end 
            f1.insert(0, f"{i} rots") # add training num
            f1.insert(0, ds_lo) # add DS left out 
            f1_scores[f'{ds_lo} training {i} ROTS'] = f1
                
            flip_preds = cnn.predict(eval_accel_aug["flips"])
            y_flip_pred = np.argmax(flip_preds, axis=1)
            y_flip_true = np.argmax(map_labels(eval_labels_aug["flips"], label_set), axis=1)
            f1 = list(f1_score(y_flip_pred, y_flip_true, average=None, labels=np.arange(output_shape)))
            f1.append(np.average(f1)) # add the average to the very end 
            f1.insert(0, f"{i} flips") # add training num
            f1.insert(0, ds_lo) # add DS left out 
            f1_scores[f'{ds_lo} training {i} FLIPS'] = f1
                
            rf_preds = cnn.predict(eval_accel_aug["rfs"])
            y_rf_pred = np.argmax(rf_preds, axis=1)
            y_rf_true = np.argmax(map_labels(eval_labels_aug["rfs"], label_set), axis=1)
            f1 = list(f1_score(y_rf_true, y_rf_pred, average=None, labels=np.arange(output_shape)))
            f1.append(np.average(f1)) # add the average to the very end 
            f1.insert(0, f"{i} rots and flips") # add training num
            f1.insert(0, ds_lo) # add DS left out 
            f1_scores[f'{ds_lo} training {i} ROTS AND FLIPS'] = f1
            
            # TODO: refactor this code to be a method
            confusion_matrices[f'{ds_lo}_training_{i}_na'] = confusion_matrix(
                label_set, 
                map_labels(eval_labels_aug["na"], label_set), preds)
            confusion_matrices[f'{ds_lo}_training_{i}_rots'] = confusion_matrix(
                label_set, 
                map_labels(eval_labels_aug["rots"], label_set), rot_preds)
            confusion_matrices[f'{ds_lo}_training_{i}_flips'] = confusion_matrix(
                label_set, 
                map_labels(eval_labels_aug["flips"], label_set), flip_preds)
            confusion_matrices[f'{ds_lo}_training_{i}_rfs'] = confusion_matrix(
                label_set, 
                map_labels(eval_labels_aug["rfs"], label_set), rf_preds)

    t2 = time.perf_counter()
    accuracies[ds_lo].append(round((t2 - t1)/3, 2))
                 
    return accuracies, f1_scores, confusion_matrices

# Train and evaluate according to the methodology of Experiment 2 in the accompanying manuscript.
# Params: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data, label_set (dict) - activities considered across 
# experiments, ds_lo (int) - the ID of the left out participant, path (string) - where loss results 
# will be written, first_training_aug (string) - the orientations to be used in the training set for 
# the first sensor, second_training_aug (string) - the orientations to be used in the training set 
# for the second sensor.
# Returns: accuracies (dict) - evaluated accuracies across evaluation sets for all seeds, f1_scores 
# (dict) - evaluated f1 scores across evaluation sets for all seeds, confusion_matrices (dict) - 
# confusion matrices across evaluation sets for all seeds. 
def exp_2_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    label_set,
    ds_lo,
    path,
    first_training_aug,
    second_training_aug):

    accuracies = {}
    f1_scores = {}
    confusion_matrices = {}
    
    first_aug_eval_labels = make_labels_for_aug_data(testing_label_sets[ds_lo])
    first_aug_eval = create_augmented_datasets(testing_accel_sets[ds_lo], 0)
    second_aug_eval_labels = make_labels_for_aug_data(first_aug_eval_labels[first_training_aug])
    second_aug_eval = create_augmented_datasets(first_aug_eval[first_training_aug], 1)
    
    # create the defined testing sets, _X is the first aug.
    aug_eval_na = create_augmented_datasets(first_aug_eval["na"], 1)
    aug_eval_rots = create_augmented_datasets(first_aug_eval["rots"], 1)
    aug_eval_flips = create_augmented_datasets(first_aug_eval["flips"], 1)
    aug_eval_rfs = create_augmented_datasets(first_aug_eval["rfs"], 1)
    aug_eval_labels_na = make_labels_for_aug_data(first_aug_eval_labels["na"])
    aug_eval_labels_rots = make_labels_for_aug_data(first_aug_eval_labels["rots"])
    aug_eval_labels_flips = make_labels_for_aug_data(first_aug_eval_labels["flips"])
    aug_eval_labels_rfs = make_labels_for_aug_data(first_aug_eval_labels["rfs"])
    
    first_aug_train_labels = make_labels_for_aug_data(training_label_sets[ds_lo])
    first_aug_train = create_augmented_datasets(training_accel_sets[ds_lo], 0)
    second_aug_train_labels = make_labels_for_aug_data(first_aug_train_labels[first_training_aug])
    second_aug_train = create_augmented_datasets(first_aug_train[first_training_aug], 1)

    accuracies[ds_lo] = [ds_lo]
    t1 = time.perf_counter()

    # concatenate training data into one array 
    if first_training_aug != "na" and second_training_aug != "na":
        train_accel = np.concatenate((
            training_accel_sets[ds_lo], 
            second_aug_train[second_training_aug]))
        train_labels = map_labels(
            np.concatenate((training_label_sets[ds_lo], second_aug_train_labels[second_training_aug])), 
            label_set)
    else:
        train_accel = training_accel_sets[ds_lo]
        train_labels = map_labels(training_label_sets[ds_lo], label_set)
    
    input_shape = train_accel.shape[1:]
    output_shape = train_labels.shape[1]
    
    # randomly initialize 3 dif models and train them 
    for i in range(1, 4): 
        tf.keras.utils.disable_interactive_logging()
        # define model 
        cnn = multi_sensor_CNN(input_shape, output_shape)
            
        # compile and train model 
        cnn.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
                
        csv_logger = CSVLogger(
            f'{path}/DS_{ds_lo}_Loss_Training_{i}.csv', 
            append=True, separator=',')
        cnn.fit(
            train_accel, 
            train_labels, 
            epochs=25,
            callbacks=[csv_logger])
                
        
        # evaluate model on evaluation sets 
        results = cnn.evaluate(aug_eval_na["na"], map_labels(aug_eval_labels_na["na"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_na["rots"], 
            map_labels(aug_eval_labels_na["rots"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_na["flips"], 
            map_labels(aug_eval_labels_na["flips"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_na["rfs"], 
            map_labels(aug_eval_labels_na["rfs"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)

        results = cnn.evaluate(
            aug_eval_rots["na"], 
            map_labels(aug_eval_labels_rots["na"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rots["rots"], 
            map_labels(aug_eval_labels_rots["rots"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rots["flips"], 
            map_labels(aug_eval_labels_rots["flips"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rots["rfs"], 
            map_labels(aug_eval_labels_rots["rfs"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_flips["na"], 
            map_labels(aug_eval_labels_flips["na"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_flips["rots"], 
            map_labels(aug_eval_labels_flips["rots"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_flips["flips"], 
            map_labels(aug_eval_labels_flips["flips"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_flips["rfs"], 
            map_labels(aug_eval_labels_flips["rfs"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rfs["na"], 
            map_labels(aug_eval_labels_rfs["na"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rfs["rots"], 
            map_labels(aug_eval_labels_rfs["rots"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rfs["flips"], 
            map_labels(aug_eval_labels_rfs["flips"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        results = cnn.evaluate(
            aug_eval_rfs["rfs"], 
            map_labels(aug_eval_labels_rfs["rfs"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
                
        # f1s are computed with the first sensor in the specified orientation
        preds = cnn.predict(second_aug_eval["na"])
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(map_labels(second_aug_eval_labels["na"], label_set), axis=1)
        f1 = list(f1_score(y_true, y_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, i) # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i}'] = f1
                
        rot_preds = cnn.predict(second_aug_eval["rots"])
        y_rot_pred = np.argmax(rot_preds, axis=1)
        y_rot_true = np.argmax(map_labels(second_aug_eval_labels["rots"], label_set), axis=1)
        f1 = list(f1_score(y_rot_true, y_rot_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} rots") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} ROTS'] = f1
                
        flip_preds = cnn.predict(second_aug_eval["flips"])
        y_flip_pred = np.argmax(flip_preds, axis=1)
        y_flip_true = np.argmax(map_labels(second_aug_eval_labels["flips"], label_set), axis=1)
        f1 = list(f1_score(y_flip_pred, y_flip_true, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} flips") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} FLIPS'] = f1
                
        rf_preds = cnn.predict(second_aug_eval["rfs"])
        y_rf_pred = np.argmax(rf_preds, axis=1)
        y_rf_true = np.argmax(map_labels(second_aug_eval_labels["rfs"], label_set), axis=1)
        f1 = list(f1_score(y_rf_true, y_rf_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} rots and flips") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} ROTS AND FLIPS'] = f1
                
        confusion_matrices[f'{ds_lo}_training_{i}_na_na'] = confusion_matrix(
            label_set, 
            map_labels(second_aug_eval_labels["na"], label_set), preds)
        confusion_matrices[f'{ds_lo}_training_{i}_rfs_rfs'] = confusion_matrix(
            label_set, 
            map_labels(second_aug_eval_labels["rfs"], label_set), rf_preds)
        
    t2 = time.perf_counter()
    accuracies[ds_lo].append(round(t2 - t1, 2))
                 
    return accuracies, f1_scores, confusion_matrices

# Train and evaluate according to the methodology of Experiment 3 in the accompanying manuscript.
# Params: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data, label_set (dict) - activities considered across 
# experiments, ds_lo (int) - the ID of the left out participant, path (string) - where loss results 
# will be written, training_aug (string) - the orientations to be used in the training set.
# Returns: accuracies (dict) - evaluated accuracies across evaluation sets for all seeds, f1_scores 
# (dict) - evaluated f1 scores across evaluation sets for all seeds, confusion_matrices (dict) - 
# confusion matrices across evaluation sets for all seeds. 
def exp_3_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    label_set,
    ds_lo,
    path,
    training_aug): #training_aug will either be na or rfs

    accuracies = {}
    f1_scores = {}
    confusion_matrices = {}

    eval_labels_aug = make_labels_for_aug_data(testing_label_sets[ds_lo])
    eval_accel_aug = create_augmented_datasets(testing_accel_sets[ds_lo], 0)
    train_labels_aug = make_labels_for_aug_data(training_label_sets[ds_lo])
    train_accel_aug = create_augmented_datasets(training_accel_sets[ds_lo], 0)
    
    accuracies[ds_lo] = [ds_lo]
    t1 = time.perf_counter()

    if training_aug != "na":
        train_accel = np.concatenate((train_accel_aug["na"], train_accel_aug[training_aug]))
        train_labels = map_labels(
            np.concatenate((train_labels_aug["na"], train_labels_aug[training_aug])), 
            label_set)
    else:
        train_accel = train_accel_aug["na"]
        train_labels = map_labels(train_labels_aug["na"], label_set)

    input_shape = train_accel[:, :, 3:].shape[1:]
    output_shape = train_labels.shape[1]

    # run 3 random seeds
    for i in range(1, 4): 
        tf.keras.utils.disable_interactive_logging()
        
        # define model 
        cnn = single_sensor_CNN(input_shape, output_shape)
            
        # compile and train model 
        cnn.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['accuracy'])
                
        csv_logger = CSVLogger(
            f'{path}/DS_{ds_lo}_Loss_Training_{i}.csv', 
            append=True, separator=',')
        cnn.fit(
            train_accel[:, :, 0:3],
            train_labels, 
            batch_size=100, 
            epochs=25,
            callbacks=[csv_logger])
        
        # evaluate model on the left out ds 
        results = cnn.evaluate(
            testing_accel_sets[ds_lo][:, :, 0:3], 
            map_labels(testing_label_sets[ds_lo], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
        
        results = cnn.evaluate(
            testing_accel_sets[ds_lo][:, :, 3:], 
            map_labels(testing_label_sets[ds_lo], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)

        results = cnn.evaluate(
            eval_accel_aug["rots"][:, :, 0:3], 
            map_labels(eval_labels_aug["rots"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)

        results = cnn.evaluate(
            eval_accel_aug["flips"][:, :, 0:3],
            map_labels(eval_labels_aug["flips"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)

        results = cnn.evaluate(
            eval_accel_aug["rfs"][:, :, 0:3], 
            map_labels(eval_labels_aug["rfs"], label_set))
        acc = round(results[1]*100, 2)
        accuracies[ds_lo].append(acc)
            
        preds = cnn.predict(testing_accel_sets[ds_lo][:, :, 0:3])
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(map_labels(testing_label_sets[ds_lo], label_set), axis=1)
                
        f1 = list(f1_score(y_true, y_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, i) # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i}'] = f1
            
        rot_preds = cnn.predict(eval_accel_aug["rots"][:, :, 0:3])
        y_rot_pred = np.argmax(rot_preds, axis=1)
        y_rot_true = np.argmax(map_labels(eval_labels_aug["rots"], label_set), axis=1)
        f1 = list(f1_score(y_rot_true, y_rot_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} rots") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} ROTS'] = f1
        
        flip_preds = cnn.predict(eval_accel_aug["flips"][:, :, 0:3])
        y_flip_pred = np.argmax(flip_preds, axis=1)
        y_flip_true = np.argmax(map_labels(eval_labels_aug["flips"], label_set), axis=1)
        f1 = list(f1_score(y_flip_pred, y_flip_true, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} flips") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} FLIPS'] = f1

        rf_preds =  cnn.predict(eval_accel_aug["rfs"][:, :, 0:3])
        y_rf_pred = np.argmax(rf_preds, axis=1)
        y_rf_true = np.argmax(map_labels(eval_labels_aug["rfs"], label_set), axis=1)
        f1 = list(f1_score(y_rf_true, y_rf_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} rots and flips") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} ROTS AND FLIPS'] = f1

        rw_preds = cnn.predict(testing_accel_sets[ds_lo][:, :, 0:3])
        y_rw_pred = np.argmax(rw_preds, axis=1)
        y_rw_true = np.argmax(map_labels(testing_label_sets[ds_lo], label_set), axis=1)
        f1 = list(f1_score(y_rw_true, y_rw_pred, average=None, labels=np.arange(output_shape)))
        f1.append(np.average(f1)) # add the average to the very end 
        f1.insert(0, f"{i} real_world") # add training num
        f1.insert(0, ds_lo) # add DS left out 
        f1_scores[f'{ds_lo} training {i} REAL WORLD'] = f1
            
        confusion_matrices[f'{ds_lo}_training_{i}_na'] = confusion_matrix(
            label_set, 
            map_labels(eval_labels_aug["na"], label_set), preds)
        confusion_matrices[f'{ds_lo}_training_{i}_RW'] = confusion_matrix(
            label_set, 
            map_labels(testing_label_sets[ds_lo], label_set), rw_preds)
        confusion_matrices[f'{ds_lo}_training_{i}_rots'] = confusion_matrix(
            label_set, 
            map_labels(eval_labels_aug["rots"], label_set), rot_preds)
        confusion_matrices[f'{ds_lo}_training_{i}_flips'] = confusion_matrix(
            label_set, 
            map_labels(eval_labels_aug["flips"], label_set), flip_preds)
        confusion_matrices[f'{ds_lo}_training_{i}_rfs'] = confusion_matrix(
            label_set, 
            map_labels(eval_labels_aug["rfs"], label_set), rf_preds)

    t2 = time.perf_counter()
    accuracies[ds_lo].append(round(t2 - t1, 2))
                 
    return accuracies, f1_scores, confusion_matrices
