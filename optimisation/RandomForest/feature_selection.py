"""
Feature selection using RFECV for Random Forest Classifier
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
import pickle
from config import *

# Converts cross validation fold data into dataframe
def convert_data_to_dataframe(selector):
    df_features = pd.DataFrame(
        columns=["feature", "support", "ranking", "mean score", "std score"]
    )
    for i in range(X_train.shape[1]):
        df_features.loc[i] = [
            selector.feature_names_in_[i],
            selector.support_[i],
            selector.ranking_[i],
            selector.cv_results_["mean_test_score"][i],
            selector.cv_results_["std_test_score"][i],
        ]
    return df_features

# Random Forest Classifier
def create_model():
    return RandomForestClassifier(n_jobs=-1)

# Import training data
X_train, Y_train = import_dataset(standardize=False)

amex_scorer = make_scorer(amex_metric_numpy, needs_proba=True)

# RFECV for Feature Selection
k_fold = StratifiedKFold(n_splits=4)
selector = RFECV(estimator=create_model(), cv=k_fold, verbose=11, scoring=amex_scorer, n_jobs=-1)
selector = selector.fit(X_train, Y_train)

dataframe_selector = convert_data_to_dataframe(selector)

# Dataframe for selected features
with open(f"{config['EXPORT']['feature_eliminated_path']}/df_selected_features", "wb") as file:
    pickle.dump(dataframe_selector, file)

selected_columns = dataframe_selector.feature.values.flatten()

# Selected Features
with open(f"{config['EXPORT']['feature_eliminated_path']}/selected_features", "wb") as file:
    pickle.dump(selected_columns, file)