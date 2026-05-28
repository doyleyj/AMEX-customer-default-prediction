"""
Feature Selection - Extraction of features from cross validation folds.
Extract fold information from pickled folds file after running L1 regression.
"""

import pickle
import numpy as np
from config import *

# Returns the average accuracy, average feature eliminated of the cross validation folds.
def extract_fold_information():
    def cv_filename(i):
        return 

    n_outer_folds = 5
    fold_infos = []
    
    avg_accuracy = None
    avg_feature_eliminated = None
    # For every outer fold get its average accuracy and average feature eliminated.
    for i in range(n_outer_folds):
        with open(f"{config['EXPORT']['cv_results_path']}/cv_results_{i}", "rb") as f:
            fold = pickle.load(f)
            fold_infos.append(fold)
            accuracy = np.array(fold["accuracy"])
            feature_eliminated = np.array(fold["num_feature_eliminated"])
            if i == 0:
                avg_accuracy = accuracy
                avg_feature_eliminated = feature_eliminated
            else:
                avg_accuracy += accuracy
                avg_feature_eliminated += feature_eliminated
                
    avg_accuracy /= n_outer_folds
    avg_feature_eliminated //= n_outer_folds
    
    # Tolerance for accuracy & feature selection.
    tolerance = 0.0007
    argmax_acc = 0
    for i, acc in enumerate(avg_accuracy):
        max_acc = avg_accuracy[argmax_acc]
        if acc > max_acc and abs(max_acc - acc) > tolerance:
            argmax_acc = i
            
    return fold_infos, avg_accuracy, avg_feature_eliminated, argmax_acc