# A Methodology for Extracting Orientation-Independent Features Using Convolutional Neural Networks to Support More Adaptable Human Activity Recognition

This repository contains all the code used in the paper "A Methodology for Extracting Orientation-Independent Features Using Convolutional Neural Networks to Support More Adaptable Human Activity Recognition". 

## System Requirements
This code is reliant on 
* [TensorFlow](https://www.tensorflow.org/install)
* [Sci-Kit Learn](https://scikit-learn.org/stable/install.html) 

All packages must be properly installed to replicate the experimental results. 

## To Replicate Experimental Results 
The script `run_exp_n.py` where *n* represents an experiment number (e.g., 1, 2, or) corresponds to the models trained and evaluated to obtain the results for experiment *n*. As the PAAWS data is not currently publically available, to replicate the results with a different dataset, modify the paths in the `get_data.py` file. This repository will be updated to work with the PAAWS data once it is publically available allowing anyone to replicate the results in this work. 

Each experiment requires a different set of arguments that are defined in the instructions below:
1. Clone this repo in the terminal using an SSH key and navigate into the directory (`cd CNN_orientation_invariance`).
2. To run Experiment 1: `python3 run_exp_1.py [dataset LO] [sensor] [training_orientation] [experiment name] [datasets to consider]`
   3. Example, to run Experiment 1 on the right wrist (top) sensor data using a training set of 5 participants datasets comprised of non-augmented and rotated orientations and leaving participant 10 out for validation: `python3 run_exp_1.py 10 "WristTop" "rots" "1_W_NA" [10, 12, 13, 114, 117, 126]`
4.  To run Experiment 2: `python3 run_exp_2.py [dataset LO] [sensor1] [sensor2] [training_orientation sensor1] [training_orientation sensor2] [experiment name] [datasets to consider]`
   5. Example, to run Experiment 2 on the right ankle (lateral) and right thigh sensor data using a training set of 5 participants datasets comprised of non-augmented (wrist) and non-augmented and flipped orientations and leaving participant 10 out for validation: `python3 run_exp_2.py 10 "AnkleLateral" "Thigh" "na" "flips" "2_WT_NA_F" [10, 27, 53, 141, 171, 226]`
6. To run Experiment 3: `python3 run_exp_3.py [dataset LO] [sensor] [training_orientation] [datasets to consider]`
   7. Example, to run Experiment 3 on the right wrist (top and bottom) sensors  using a training set of 5 participants datasets comprised solely of non-augmented data and leaving participant 10 out for validation: `python3 run_exp_3.py 10 "WristTop" "WristBottom "na" "3_W_NA" [10, 28, 33, 76, 100, 113]`

Note: these instructions assume `get_data.py` has been modified accordingly. Further, the training orientation argument denotes the orientations to be included in the training data in addition to the non-augmented (original) data.  
