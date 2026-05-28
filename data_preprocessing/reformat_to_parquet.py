"""
One hot encoding is performed.
Export to parquet.
"""
import gc
import pandas as pd

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

# Categorical features to OHE.
categorical_features = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]

# ---------------------------------------------------------------------------- #
#                              Data Preprocessing                              #
# ---------------------------------------------------------------------------- #


def preprocess_data(X):
    # Drop date column.
    X.drop("S_2", axis=1, inplace=True)


# ---------------------------------------------------------------------------- #
#                               Training Dataset                               #
# ---------------------------------------------------------------------------- #
print("\n# ----- Preprocessing Train Dataset ----- #")
print("Read train_data.csv.")
X_train = pd.read_csv(config["DATAPATH"]["Raw_Train_Path"])
preprocess_data(X_train)
print("Training shape (Before OHE): ", X_train.shape)

# Perform OHE.
print("OHE train_data")
X_train = pd.get_dummies(X_train, columns=categorical_features)

# Export to parquet.
print("Export train_data")
X_train.to_parquet(config["DATAPATH"]["Parquet_Train_Path"])

print("Training shape (After OHE): ", X_train.shape)

training_columns = X_train.columns

del X_train
gc.collect()

# ---------------------------------------------------------------------------- #
#                                Testing Dataset                               #
# ---------------------------------------------------------------------------- #
print("\n# ----- Preprocessing Test Dataset ----- #")
print("Read test_data.csv.")
X_test = pd.read_csv(config["DATAPATH"]["Raw_Test_Path"])
preprocess_data(X_test)
print("Testing shape (Before OHE): ", X_test.shape)

# Perform OHE.
print("OHE test_data")
X_test = pd.get_dummies(X_test, columns=categorical_features)
# Re-index test columns after OHE.
X_test = X_test.reindex(columns=training_columns, fill_value=0)

# Export to parquet.
print("Export test_data")
X_test.to_parquet(config["DATAPATH"]["Parquet_Test_Path"])

print("Testing shape (After OHE): ", X_test.shape)

del X_test
gc.collect()
