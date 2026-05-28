"""
Import Data
"""
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from configparser import ConfigParser

config_path = "../../config.ini"

# Imports Train/Test Dataset. Pass feature_selection_file_path for feature selection.
def import_dataset(feature_selection_file_path=None, is_feature_dataframe=False, eliminate=True, standardize=True, is_feature_set_1=False, is_test=False):
    # Load Configuration File
    config = ConfigParser()
    config.read(config_path)
    
    if is_test:
        print("### Test Dataset ###")
    else:
        print("### Train Dataset ###")
    
    # Import Data
    print("data importing..")
    if is_test:
        if is_feature_set_1:
            X = pd.read_parquet(config["DATAPATH"]["Test_FeatureSet1_Path"])
        else:
            X = pd.read_parquet(config["DATAPATH"]["Test_FeatureSet2_Path"])
        X_index = X.index
    else:
        if is_feature_set_1:
            X = pd.read_parquet(config["DATAPATH"]["Train_FeatureSet1_Path"])
        else:
            X = pd.read_parquet(config["DATAPATH"]["Train_FeatureSet2_Path"])
        Y = pd.read_csv(config["DATAPATH"]["Train_Label_Path"]).target
    print("data imported")

    # Feature Selection
    if feature_selection_file_path is not None:
        print("feature selection..")
        if is_feature_dataframe:
            feature_ranking = pd.read_csv(feature_selection_file_path)
            selected_columns = feature_ranking.feature[feature_ranking.support == True].values
        else:
            with open(feature_selection_file_path, "rb") as file:
                selected_columns = pickle.load(file)
            if eliminate:
                selected_columns = list(set(X.columns).difference(set(selected_columns)))
        X = X[selected_columns]
        print("feature selected")

    # Standardize Data
    if standardize:
        print("data standardizing..")
        customer_index = X.index
        feature_name_columns = X.columns
        X = pd.DataFrame(StandardScaler().fit_transform(X))
        X.columns = feature_name_columns
        X.sort_index(axis=1, inplace=True)
        X.set_index(customer_index, inplace=True)
        print("data standardized")
    X.fillna(X.mean(), inplace=True)

    if is_test:
        return X, X_index
    else:
        return X, Y