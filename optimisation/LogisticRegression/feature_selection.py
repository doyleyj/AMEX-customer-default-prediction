"""
Feature Selection with Nested Cross Validation
Use Logistic Regression L1 to eliminate features that have zero coefficients.
"""

import numpy as np
import pickle
import time
from sklearn.model_selection import KFold, StratifiedKFold
from cuml import LogisticRegression
from config import *

# The Logistic Regression Model
def create_model(regularization_strength):
    return LogisticRegression(penalty="l1", max_iter=3000, C=regularization_strength)

# Nested Cross Validation
outer_cv_folds, inner_cv_folds = 5, 4
outer_folds = StratifiedKFold(n_splits=outer_cv_folds)
inner_folds = KFold(n_splits=inner_cv_folds)

# Feature Selection with L1
C_grid = np.linspace(0.00002, 0.2, 40)

# Import training data
X_train, Y_train = import_dataset()

# Outer CV Loop - Model evaluation
for fold, (i_train, i_test) in enumerate(outer_folds.split(X_train, Y_train)):
    print(f"\nFold (Outer CV): {fold} , Started!")
    
    # Measures time per outer cv loop.
    start = time.time()
    
    X_train, X_test = X_train.iloc[i_train], X_train.iloc[i_test]
    Y_train, Y_test = Y_train[i_train], Y_train[i_test]
    
    # Inner CV Loop - Feature selection with L1
    for fold_inner, (j_train, j_validation) in enumerate(inner_folds.split(X_train, Y_train)):
        print(f"Fold (Inner CV): {fold_inner}")
        
        X_train_in, X_validate_in = X_train.iloc[j_train], X_train.iloc[j_validation]
        Y_train_in, Y_validate_in = Y_train.iloc[j_train], Y_train.iloc[j_validation]
        
        fold_info = {"c_grid": C_grid, "num_feature_eliminated": [], 
                    "feature_eliminated": [], "accuracy": []}
        
        best_accuracy = 0
        for c in C_grid:
            model = create_model(c)
            model.fit(X_train_in, Y_train_in.values)
            predictions = model.predict_proba(X_validate_in)[:, 1]
            accuracy = amex_metric_numpy(Y_validate_in.values, predictions)
            
            # Extract fold information for the current model.
            feature_eliminated = [feature for feature, coef in zip(X_train.columns, model.coef_) if coef == 0]
            fold_info["num_feature_eliminated"].append(len(feature_eliminated))
            fold_info["feature_eliminated"].append(feature_eliminated)
            fold_info["accuracy"].append(accuracy)
            record = f"[C={round(c, 6)}] Accuracy={round(accuracy, 3)} TotalFeatureEliminated={len(feature_eliminated)}"
            print(record)
            if accuracy > best_accuracy:
                best_accuracy, best_record = accuracy, record
                
        print(f"Best Search: {best_record}\n")
        
        # Export fold information for plots etc.
        with open(f"{config['EXPORT']['cv_results_path']}/cv_results_{fold}", "wb") as file:
            pickle.dump(fold_info, file)
    
    print(f"Time taken for fold {fold}: {round((time.time()-start)/60, 2)} minutes.")