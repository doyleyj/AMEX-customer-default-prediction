from xgboost import XGBClassifier
import sys
sys.path.append('../')
from stack_train import train_model
sys.path.append('../../helper_functions')
from amex_metric import amex_metric
from importer import import_dataset

def create_model():
    return XGBClassifier(n_estimators=988, booster="gbtree", reg_lambda=9.578603750508189e-08,
                         reg_alpha=0.0002776830223437006, max_depth=6, learning_rate=0.028141902625074152,
                         gamma=5.99773230664579e-06, grow_policy="lossguide",
                         objective="binary:logistic", disable_default_eval_metric=1, tree_method="gpu_hist",
                         eval_metric=amex_metric)

feature_selection_file = "XGB_feature_ranking.csv"
X_train, Y_train = import_dataset(feature_selection_file_path=feature_selection_file, is_feature_dataframe=True, eliminate=False, standardize=False)
X_test, X_index = import_dataset(feature_selection_file_path=feature_selection_file, is_feature_dataframe=True, eliminate=False, standardize=False, is_test=True)

model = create_model()
train_predictions, Y_vals, test_predictions = train_model(model, X_train, Y_train, X_test)

train_predictions.to_csv("stacking_train_pred.csv")
test_predictions.to_csv("stacking_test_pred.csv")