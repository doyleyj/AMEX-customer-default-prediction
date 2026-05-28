import pandas as pd
import xgboost as xgb
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
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "max_depth" : trial.suggest_int("max_depth", 1, 9),
        "learning_rate" : trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "gamma" : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "grow_policy" : trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }
    scorer = make_scorer(amex_metric_numpy, greater_is_better=True, needs_proba=True)
    model = xgb.XGBClassifier(**tune_parameters,
                              tree_method="gpu_hist")
    cv = StratifiedKFold(n_splits=4)
    accuracy = cross_val_score(model, X=train_features.values, y=train_labels.values, cv=cv, scoring=scorer)
    return accuracy.mean()


print("data importing..")
X_train = pd.read_parquet("../../data_preprocessing/X_train_feature_set_2.parquet")
Y_train = pd.read_csv("../../data_preprocessing/train_labels.csv").target
print("data imported")
feature_ranking = pd.read_csv("XGB_feature_ranking.csv")
optimal_features = feature_ranking.feature[feature_ranking.support == True].values
X_train = X_train[optimal_features]
print("Sub-optimal features dropped")
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, train_features=X_train, train_labels=Y_train), n_trials=2)
print("Best trial:")
trial = study.best_trial
print(trial)

joblib.dump(study, "XGB_study_feature_set_2.pkl")
print("Study Exported")
figure = optuna.visualization.plot_param_importances(study)
figure.write_image("XGB_param_importances_feature_set_2.png")
figure1 = optuna.visualization.plot_parallel_coordinate(study)
figure1.write_image("XGB_parallel_coordinate_feature_set_2.png")
figure2 = optuna.visualization.plot_optimization_history(study)
figure2.write_image("XGB_optimization_history_feature_set_2.png")
