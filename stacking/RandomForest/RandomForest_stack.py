from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('../')
from stack_train import train_model
sys.path.append('../../helper_functions')
from importer import import_dataset

def create_model():
    return RandomForestClassifier(n_estimators=927, max_depth=77, min_samples_split=8, min_samples_leaf=4, n_jobs=-1)

feature_selection_file = "selected_features"
X_train, Y_train = import_dataset(feature_selection_file_path=feature_selection_file, eliminate=False, standardize=False)
X_test, X_index = import_dataset(feature_selection_file_path=feature_selection_file, eliminate=False, standardize=False, is_test=True)

model = create_model()
train_predictions, Y_vals, test_predictions = train_model(model, X_train, Y_train, X_test)

train_predictions.to_csv("stacking_train_pred.csv")
test_predictions.to_csv("stacking_test_pred.csv")