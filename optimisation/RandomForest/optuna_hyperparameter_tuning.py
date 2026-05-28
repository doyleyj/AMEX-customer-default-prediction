"""
Parameter tuning with Optuna for Random Forest Classifier.
"""

import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
from functools import partial
from sklearn.metrics import make_scorer
from config import *

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

# Objective function to maximize
def objective(trial, train_features, train_labels, scorer):
    # Parameters to tune
    _n_estimators = trial.suggest_int("n_estimators", 2, 1200)
    _max_depth = trial.suggest_int("max_depth", 10, 100)
    _min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    model = RandomForestClassifier(n_estimators=_n_estimators, max_depth=_max_depth, min_samples_leaf=_min_samples_leaf, min_samples_split=_min_samples_split, n_jobs=-1)
    cv = StratifiedKFold(n_splits=4)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring=scorer, n_jobs=-1)
    avg_score = np.mean(cv_scores)
    
    return avg_score

# Import training data
X_train, Y_train = import_dataset(f"{config['EXPORT']['feature_eliminated_path']}/selected_features", eliminate=False, standardize=False)

# Amex Scoring
amex_scorer = make_scorer(amex_metric_numpy, needs_proba=True)

# Optuna - Optimize parameters
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, train_features=X_train, train_labels=Y_train, scorer=amex_scorer), n_trials=100)

# Export study
with open(f"{config['EXPORT']['optuna_study_path']}/study", "wb") as file:
    pickle.dump(study, file)

# Export optuna plots
figure = optuna.visualization.plot_param_importances(study)
figure.write_image(f"{config['EXPORT']['image_path']}/rf_param_importances.png")
figure = optuna.visualization.plot_optimization_history(study)
figure.write_image(f"{config['EXPORT']['image_path']}/rf_optimization_history.png")
figure = optuna.visualization.plot_parallel_coordinate(study)
figure.write_image(f"{config['EXPORT']['image_path']}/rf_parallel_coordinate.png")
    
# The best studies / parameters tuned
print(len(study.trials))
print("Best trial:")
trial = study.best_trial
print(trial.value)
for k, v in trial.params.items():
    print(f"{k} : {v}")
