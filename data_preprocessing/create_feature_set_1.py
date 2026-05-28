import pandas as pd

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

# Train Parquet - Create Feature Set 1 and save as X_train_feature_set_1.parquet
X_train = pd.read_parquet(config["DATAPATH"]["Parquet_Train_Path"])
X_train = X_train.groupby("customer_ID").last().sort_index() # Simply extract the last customer record.
X_train.fillna((X_train.mean()), inplace=True) # Fill N/A values with mean.
X_train.to_parquet(config["DATAPATH"]["Train_FeatureSet1_Path"])

# Test Parquet - Create Feature Set 1 and save as X_test_feature_set_1.parquet
X_test = pd.read_parquet(config["DATAPATH"]["Parquet_Test_Path"])
X_test = X_test.groupby("customer_ID").last().sort_index() # Simply extract the last customer record.
X_test.fillna((X_test.mean()), inplace=True) # Fill N/A values with mean.
X_test.to_parquet(config["DATAPATH"]["Test_FeatureSet1_Path"])
