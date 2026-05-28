from cuml import LogisticRegression
import sys
sys.path.append('../')
from stack_train import train_model
sys.path.append('../../helper_functions')
from importer import import_dataset

def create_model():
    return LogisticRegression(C=0.03233598851589464, max_iter=1463, penalty="l1", linesearch_max_iter=161)

feature_selection_file = "L1_eliminated_features.pkl"
X_train, Y_train = import_dataset(feature_selection_file_path=feature_selection_file)
X_test, X_index = import_dataset(feature_selection_file_path=feature_selection_file, is_test=True)

model = create_model()
train_predictions, Y_values, test_predictions = train_model(model, X_train, Y_train, X_test)

train_predictions.to_csv("stacking_train_pred.csv")
test_predictions.to_csv("stacking_test_pred.csv")
Y_values.to_csv("stacking_Y_train.csv", index=False)