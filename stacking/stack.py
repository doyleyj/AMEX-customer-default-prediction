import pandas as pd
from cuml import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from stack_train import train_model
from sklearn.metrics import make_scorer
import sys
sys.path.append('../helper_functions')
from amex_metric import amex_metric_numpy
from functools import partial
import optuna
import numpy as np
import pickle

# ------------------------------------------------------------------ #
#                          Model Predictions                         #
# ------------------------------------------------------------------ #
train_pred_path = "stacking_train_pred.csv"
test_pred_path = "stacking_test_pred.csv"

# XGBoost Predictions
xgb_train_preds = pd.read_csv(f"XGBoost/{train_pred_path}")["prediction"]
xgb_test_preds = pd.read_csv(f"XGBoost/{test_pred_path}").iloc[:, 1:].mean(axis=1)

# LightGBM Predictions
lgbm_train_preds = pd.read_csv(f"LightGBM/{train_pred_path}")["prediction"]
lgbm_test_preds = pd.read_csv(f"LightGBM/{test_pred_path}").iloc[:, 1:].mean(axis=1)

# Logistic Regression Predictions
lr_train_preds = pd.read_csv(f"LogisticRegression/{train_pred_path}")["prediction"]
lr_test_preds = pd.read_csv(f"LogisticRegression/{test_pred_path}").iloc[:, 1:].mean(axis=1)

# Random Forest Predictions
rf_train_preds = pd.read_csv(f"RandomForest/{train_pred_path}")["prediction"]
rf_test_preds = pd.read_csv(f"RandomForest/{test_pred_path}").iloc[:, 1:].mean(axis=1)

# ------------------------------------------------------------------ #
#                        Train / Test Dataset                        #
# ------------------------------------------------------------------ #
# X_train (XGBoost, LightGBM, Logistic Regression, Random Forest Classifier)
X_train = pd.concat([xgb_train_preds, lgbm_train_preds, lr_train_preds, rf_train_preds], axis=1)

# Y_train
Y_train = pd.read_csv(f"LogisticRegression/stacking_Y_train.csv")

# X_test
X_test = pd.concat([xgb_test_preds, lgbm_test_preds, lr_test_preds, rf_test_preds], axis=1)

# Customer Test Index
X_index = pd.read_csv(f"LogisticRegression/{test_pred_path}").iloc[:, 0]

# ------------------------------------------------------------------ #
#                              Stacking                              #
# ------------------------------------------------------------------ #

# Model Correlation
print(X_test.corr())

OPTIMIZE_STACK = False

if OPTIMIZE_STACK:
    # Objective function to maximize
    def objective(trial, train_features, train_labels, scorer):
        # Parameters to tune
        _C = trial.suggest_float("C", 0.00002, 0.2)
        _max_iter = trial.suggest_int("max_iter", 1000, 2000)
        _penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        _linesearch_max_iter = trial.suggest_int("linesearch_max_iter", 70, 200)
        
        model = LogisticRegression(penalty=_penalty, max_iter=_max_iter, C=_C, linesearch_max_iter=_linesearch_max_iter)
        cv = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, train_features, train_labels.values, cv=cv, scoring=scorer)
        avg_score = np.mean(cv_scores)
        
        return avg_score
    
    # Amex Scoring
    amex_scorer = make_scorer(amex_metric_numpy, needs_proba=True)
    
    study = optuna.create_study(direction="maximize")
    # Optuna - Optimize parameters
    study.optimize(partial(objective, train_features=X_train, train_labels=Y_train.iloc[:,0], scorer=amex_scorer), n_trials=100)
    
    # Export optuna plots
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_image(f"param_importances.png")
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_image(f"optimization_history.png")
    figure = optuna.visualization.plot_parallel_coordinate(study)
    figure.write_image(f"parallel_coordinate.png")
        
    # The best studies / parameters tuned
    print(len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(trial.value)
    for k, v in trial.params.items():
        print(f"{k} : {v}")

# Stacking
stack_model = LogisticRegression(C=0.0003026774436717798, max_iter=1456, penalty="l1", linesearch_max_iter=114)
_, _, test_pred = train_model(stack_model, X_train, Y_train, X_test)
predictions = test_pred.mean(axis=1)
submission = pd.DataFrame({"customer_ID": X_index, "prediction": predictions})
submission.to_csv(f"submission_stacked_model.csv", index=False)