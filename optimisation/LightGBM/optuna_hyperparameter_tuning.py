import pandas as pd
import lightgbm as lgb
import optuna
from functools import partial
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from sklearn.metrics import make_scorer
import sys
sys.path.append('../../helper_functions')
from amex_metric import amex_metric_numpy


def objective(trial, train_features, train_labels):
    tune_parameters = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        "n_estimators": trial.suggest_int("n_estimators", 600, 1300),
        "num_leaves": trial.suggest_int("num_leaves", 5, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 60, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 60, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 250),
    }

    scorer = make_scorer(amex_metric_numpy, greater_is_better=True, needs_proba=True)
    model = lgb.LGBMClassifier(**tune_parameters)
    cv = StratifiedKFold(n_splits=4)
    accuracy = cross_val_score(model, X=train_features.values, y=train_labels.values, cv=cv, scoring=scorer)
    return accuracy.mean()


print("data importing..")
X_train = pd.read_parquet("../../data_preprocessing/X_train_feature_set_2.parquet")
Y_train = pd.read_csv("../../data_preprocessing/train_labels.csv").target
print("data imported")
feature_ranking = pd.read_csv("LightGBM_feature_ranking.csv")
optimal_features = feature_ranking.feature[feature_ranking.support == True].values
X_train = X_train[optimal_features]
print("Sub-optimal features dropped")
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, train_features=X_train, train_labels=Y_train), n_trials=100)
print("Best trial:")
trial = study.best_trial
print(trial)

joblib.dump(study, "LightGBM_study_feature_set_2.pkl")
print("Study Exported")
figure = optuna.visualization.plot_param_importances(study)
figure.write_image("LightGBM_param_importances_feature_set_2.png")
figure1 = optuna.visualization.plot_parallel_coordinate(study)
figure1.write_image("LightGBM_parallel_coordinate_feature_set_2.png")
figure2 = optuna.visualization.plot_optimization_history(study)
figure2.write_image("LightGBM_optimization_history_feature_set_2.png")
