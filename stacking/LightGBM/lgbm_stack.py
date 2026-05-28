import lightgbm as lgb
import sys
sys.path.append('../')
from stack_train import train_model
sys.path.append('../../helper_functions')
from importer import import_dataset

def create_model():
    return lgb.LGBMClassifier(n_estimators=871, num_leaves=160, learning_rate=0.017732096148663643,
    lambda_l1=9.94111878192181e-08, lambda_l2=10.875217890395598, feature_fraction=0.5428576791882828,
    bagging_fraction=0.8422136374521029, bagging_freq=5, min_child_samples=228, device="gpu")
    
feature_selection_file = "LightGBM_feature_ranking.csv"
X_train, Y_train = import_dataset(feature_selection_file_path=feature_selection_file, is_feature_dataframe=True, eliminate=False, standardize=False)
X_test, X_index = import_dataset(feature_selection_file_path=feature_selection_file, is_feature_dataframe=True, eliminate=False, standardize=False, is_test=True)

model = create_model()
train_predictions, Y_vals, test_predictions = train_model(model, X_train, Y_train, X_test)

train_predictions.to_csv("stacking_train_pred.csv")
test_predictions.to_csv("stacking_test_pred.csv")