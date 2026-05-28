"""
Get the features to be removed.
"""

import pickle
from collections import Counter
from feature_selection_extract import extract_fold_information
from config import *

fold_infos, avg_accuracy, avg_feature_eliminated, argmax_acc = extract_fold_information()

# Every fold aggregate features which has been commonly eliminated at best accuracy point.
features = Counter()
for fold in fold_infos:
    features += Counter(fold["feature_eliminated"][argmax_acc])
    
features_eliminated = [f for f, _ in features.most_common(avg_feature_eliminated[argmax_acc])]

with open(config['EXPORT']['l1_feature_eliminated_path'], "wb") as file:
    pickle.dump(features_eliminated, file)