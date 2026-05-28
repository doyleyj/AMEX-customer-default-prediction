"""
Parameter tuning with Optuna for Logistic Regression.
"""

import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from cuml import LogisticRegression
from functools import partial
import optuna
from config import *

# Objective function to maximize
def objective(trial, train_features, train_labels, scorer):
    # Parameters to tune
    _C = trial.suggest_float("C", 0.00002, 0.2)
    _max_iter = trial.suggest_int("max_iter", 1000, 2000)
    _penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    _linesearch_max_iter = trial.suggest_int("linesearch_max_iter", 70, 200)
    
    model = LogisticRegression(penalty=_penalty, max_iter=_max_iter, C=_C, linesearch_max_iter=_linesearch_max_iter)
    cv = StratifiedKFold(n_splits=4)
    cv_scores = cross_val_score(model, train_features, train_labels.values, cv=cv, scoring=scorer)
    avg_score = np.mean(cv_scores)
    
    return avg_score

# Import training data
X_train, Y_train = import_dataset(config["EXPORT"]["l1_feature_eliminated_path"], True)

# Amex Scoring
amex_scorer = make_scorer(amex_metric_numpy, needs_proba=True)

# Optuna - Optimize parameters
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, train_features=X_train, train_labels=Y_train, scorer=amex_scorer), n_trials=200)

# Export study
with open(config["EXPORT"]["optuna_study_path"], "wb") as file:
    pickle.dump(study, file)

# Export optuna plots
figure = optuna.visualization.plot_param_importances(study)
figure.write_image(f"{config['EXPORT']['image_path']}/logreg_param_importances.png")
figure = optuna.visualization.plot_optimization_history(study)
figure.write_image(f"{config['EXPORT']['image_path']}/logreg_optimization_history.png")
figure = optuna.visualization.plot_parallel_coordinate(study)
figure.write_image(f"{config['EXPORT']['image_path']}/logreg_parallel_coordinate.png")
    
# The best studies / parameters tuned
print(len(study.trials))
print("Best trial:")
trial = study.best_trial
print(trial.value)
for k, v in trial.params.items():
    print(f"{k} : {v}")
