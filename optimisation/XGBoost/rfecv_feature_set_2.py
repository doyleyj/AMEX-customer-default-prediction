import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV, RFE

import sys
sys.path.append('../../helper_functions')
from amex_metric import amex_metric_numpy

def model():
    return XGBClassifier(n_jobs=-1, n_estimators=100)

def plot_feature_selection(selectors):
    plt.figure()
    plt.title("Cross Validated AMEX Score vs. Number of Selected Features")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross Validated AMEX Score")
    for i in range(len(selectors)):
        scores_mean = selectors[i].cv_results_["mean_test_score"]
        n_features = scores_mean.shape[0]
        x_range = np.linspace(1, n_features, n_features)
        plt.plot(x_range, scores_mean)
    plt.savefig("XGB_cv_scores.png")


def save_data_to_csv(df_features):
    df_features.to_csv(f"XGB_feature_ranking.csv")


def convert_data_to_dataframe(selector):
    df_features = pd.DataFrame(
        columns=["feature", "support", "ranking", "mean score", "std score"]
    )
    for i in range(features_df.shape[1]):
        df_features.loc[i] = [
            selector.feature_names_in_[i],
            selector.support_[i],
            selector.ranking_[i],
            selector.cv_results_["mean_test_score"][i],
            selector.cv_results_["std_test_score"][i],
        ]
    return df_features

X_train = pd.read_parquet("../../data_preprocessing/X_train_feature_set_2.parquet")
Y_train = pd.read_csv("../../data_preprocessing/train_labels.csv")
features = list(X_train.columns)[:10]
amex_scorer = make_scorer(amex_metric_numpy, needs_proba=True)
print("data imported")

features_df = X_train.iloc[:][features]
if features_df.isnull().values.any():
    features_df.fillna(features_df.mean(), inplace=True)

Y_train.set_index("customer_ID", inplace=True)
Y_train = Y_train.values.reshape((-1,))

tree_model = model()
selector = RFECV(
    estimator=tree_model,
    cv=4,
    verbose=11,
    scoring=amex_scorer,
    n_jobs=-1,
)
selector = selector.fit(features_df, Y_train)
rfecv_result_df = convert_data_to_dataframe(selector)
rfecv_result_df = rfecv_result_df.sort_values(by="ranking")
save_data_to_csv(rfecv_result_df)
plot_feature_selection([selector])
