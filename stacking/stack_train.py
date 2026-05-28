import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 5 Stratified KFold for stacking per model.
def train_model(model, train, train_class, test):
    k_fold = StratifiedKFold(n_splits=5)
    train_pred = pd.DataFrame()
    test_preds = pd.DataFrame()
    cust_index = pd.DataFrame(list(test.index), columns=["customer_ID"])
    test_preds = pd.concat([test_preds, cust_index], axis=0, copy=False)
    all_preds = pd.DataFrame()
    Y_values = pd.DataFrame()
    for fold, (i_train, i_validation) in enumerate(k_fold.split(train, train_class)):
        print(f"(Predictions For Stacking) Fold: {fold}")
        X_tr = train.iloc[i_train]
        X_va = train.iloc[i_validation]
        Y_tr = train_class.iloc[i_train]
        Y_va = train_class.iloc[i_validation]
        
        model.fit(X_tr.values, Y_tr.values)
        cust_index = pd.DataFrame(list(X_va.index), columns=["customer_ID"])
        train_pred = pd.concat([train_pred, cust_index], axis=0, copy=False)
        tree_y_va_pred = model.predict_proba(X_va.values)[:, -1]
        preds = pd.DataFrame(tree_y_va_pred, columns=["prediction"])
        all_preds = pd.concat([all_preds, preds], axis=0, copy=False)
        Y_values = pd.concat([Y_values, Y_va], axis=0, copy=False)
        tree_test_pred = model.predict_proba(test.values)[:, -1]
        test_pred = pd.DataFrame(tree_test_pred, columns=["prediction"])
        test_preds = pd.concat([test_preds, test_pred], axis=1, copy=False)
        
    train_pred = pd.concat([train_pred, all_preds], axis=1, copy=False)
    train_pred.set_index("customer_ID", inplace=True)
    test_preds.set_index("customer_ID", inplace=True)
    return train_pred, Y_values, test_preds