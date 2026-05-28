import pandas as pd
import numpy as np

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

X_train_path = config["DATAPATH"]["Train_Data_Path"]
Y_train_path = config["DATAPATH"]["Train_Label_Path"]
correlation_path = config["DATAPATH"]["Correlation_Path"]

# Read Training Data and Labels
print(f"Reading in {X_train_path} file")
X_train = pd.read_parquet(X_train_path)
print(f"File read, dropping date columns and putting mean of column into nan")
Y_train = pd.read_csv(Y_train_path)
Y_train.set_index("customer_ID", inplace=True)
print("Date column removed")

feature_dict = dict()
print("Adding blank feature scores")
for feature in X_train.columns[1:]:
    feature_dict[feature] = (0, 0, "last")  # (Correlation, Period, Method)
print("Complete")


feature_set_1_cols = [
    # numerical feature
    "P_2",
    "D_39",
    "B_1",
    "B_2",
    "R_1",
    "S_3",
    "D_41",
    "B_3",
    "D_42",
    "D_43",
    "D_44",
    "B_4",
    "D_45",
    "B_5",
    "R_2",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "B_6",
    "B_7",
    "B_8",
    "D_50",
    "D_51",
    "B_9",
    "R_3",
    "D_52",
    "P_3",
    "B_10",
    "D_53",
    "S_5",
    "B_11",
    "S_6",
    "D_54",
    "R_4",
    "S_7",
    "B_12",
    "S_8",
    "D_55",
    "D_56",
    "B_13",
    "R_5",
    "D_58",
    "S_9",
    "B_14",
    "D_59",
    "D_60",
    "D_61",
    "B_15",
    "S_11",
    "D_62",
    "D_65",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_20",
    "S_12",
    "R_6",
    "S_13",
    "B_21",
    "D_69",
    "B_22",
    "D_70",
    "D_71",
    "D_72",
    "S_15",
    "B_23",
    "D_73",
    "P_4",
    "D_74",
    "D_75",
    "D_76",
    "B_24",
    "R_7",
    "D_77",
    "B_25",
    "B_26",
    "D_78",
    "D_79",
    "R_8",
    "R_9",
    "S_16",
    "D_80",
    "R_10",
    "R_11",
    "B_27",
    "D_81",
    "D_82",
    "S_17",
    "R_12",
    "B_28",
    "R_13",
    "D_83",
    "R_14",
    "R_15",
    "D_84",
    "R_16",
    "B_29",
    "S_18",
    "D_86",
    "D_87",
    "R_17",
    "R_18",
    "D_88",
    "B_31",
    "S_19",
    "R_19",
    "B_32",
    "S_20",
    "R_20",
    "R_21",
    "B_33",
    "D_89",
    "R_22",
    "R_23",
    "D_91",
    "D_92",
    "D_93",
    "D_94",
    "R_24",
    "R_25",
    "D_96",
    "S_22",
    "S_23",
    "S_24",
    "S_25",
    "S_26",
    "D_102",
    "D_103",
    "D_104",
    "D_105",
    "D_106",
    "D_107",
    "B_36",
    "B_37",
    "R_26",
    "R_27",
    "D_108",
    "D_109",
    "D_110",
    "D_111",
    "B_39",
    "D_112",
    "B_40",
    "S_27",
    "D_113",
    "D_115",
    "D_118",
    "D_119",
    "D_121",
    "D_122",
    "D_123",
    "D_124",
    "D_125",
    "D_127",
    "D_128",
    "D_129",
    "B_41",
    "B_42",
    "D_130",
    "D_131",
    "D_132",
    "D_133",
    "R_28",
    "D_134",
    "D_135",
    "D_136",
    "D_137",
    "D_138",
    "D_139",
    "D_140",
    "D_141",
    "D_142",
    "D_143",
    "D_144",
    "D_145",
    # categorical features
    "B_30_0.0",
    "B_30_1.0",
    "B_30_2.0",
    "B_38_1.0",
    "B_38_2.0",
    "B_38_3.0",
    "B_38_4.0",
    "B_38_5.0",
    "B_38_6.0",
    "B_38_7.0",
    "D_114_0.0",
    "D_114_1.0",
    "D_116_0.0",
    "D_116_1.0",
    "D_117_-1.0",
    "D_117_1.0",
    "D_117_2.0",
    "D_117_3.0",
    "D_117_4.0",
    "D_117_5.0",
    "D_117_6.0",
    "D_120_0.0",
    "D_120_1.0",
    "D_126_-1.0",
    "D_126_0.0",
    "D_126_1.0",
    "D_63_CL",
    "D_63_CO",
    "D_63_CR",
    "D_63_XL",
    "D_63_XM",
    "D_63_XZ",
    "D_64_-1",
    "D_64_O",
    "D_64_R",
    "D_64_U",
    "D_66_0.0",
    "D_66_1.0",
    "D_68_0.0",
    "D_68_1.0",
    "D_68_2.0",
    "D_68_3.0",
    "D_68_4.0",
    "D_68_5.0",
    "D_68_6.0",
]

periods = [1, 3, 6, 13]
methods = [
    "first",
    "last-first",
    "last/first",
    "mean",
    "min",
    "max",
    "std",
    "sum",
    "median",
    # log features
    "log_first",
    "log_last-first",
    "log_last/first",
    "log_mean",
    "log_min",
    "log_max",
    "log_std",
    "log_sum",
    "log_median",
]

for i in range(len(feature_set_1_cols)):

    print(f"Exploring column: {feature_set_1_cols[i]}")

    for period in periods:

        # get data for the period with customer_id and column
        cols_array = ["customer_ID"] + [feature_set_1_cols[i]]
        period_last_entries_for_customers = X_train.groupby(["customer_ID"])[
            cols_array
        ].tail(period)

        if (
            period_last_entries_for_customers[feature_set_1_cols[i]]
            .isnull()
            .values.any()
        ):
            period_last_entries_for_customers[feature_set_1_cols[i]].fillna(
                (period_last_entries_for_customers[feature_set_1_cols[i]].mean()),
                inplace=True,
            )

        for method in methods:

            log_feature = False
            method = method.split("_")
            if len(method) > 1:
                log_feature = True
            method = method[-1]

            # group by customer ID and find means
            if method == "std":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).std()
            elif method == "squ":
                new_feature = (
                    period_last_entries_for_customers.groupby(["customer_ID"]).last()
                    ** 2
                )
            elif method == "median":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).median()
            elif method == "min":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).min()
            elif method == "max":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).max()
            elif method == "sum":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).sum()
            elif method == "product":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).prod()
            elif method == "first":
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).first()
            elif method == "last-first":
                first = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).first()
                last = period_last_entries_for_customers.groupby(["customer_ID"]).last()
                new_feature = last.subtract(first)
            elif method == "last/first":
                first = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).first()
                last = period_last_entries_for_customers.groupby(["customer_ID"]).last()
                new_feature = last.divide(first)
                if np.isinf(new_feature.values).any():
                    continue
            else:
                new_feature = period_last_entries_for_customers.groupby(
                    ["customer_ID"]
                ).mean()

            if log_feature and (new_feature[feature_set_1_cols[i]] > 0).all():
                new_feature[feature_set_1_cols[i]] = np.log2(
                    new_feature[feature_set_1_cols[i]]
                )
                method = f"log_{method}"
            elif log_feature:
                continue

            new_feature_name = f"{feature_set_1_cols[i]}_{period}_periods_{method}"
            new_feature.rename(
                columns={feature_set_1_cols[i]: new_feature_name}, inplace=True
            )

            new_feature = new_feature.join(Y_train["target"])

            correlation = new_feature.corr()["target"].iloc[0]

            old_correlation, old_period, old_method = feature_dict[
                feature_set_1_cols[i]
            ]
            if abs(correlation) > abs(old_correlation):
                print(f"{new_feature_name}: {correlation}")
                feature_dict[feature_set_1_cols[i]] = (correlation, period, method)

# Export correlation information to file.
f = open(correlation_path, "w")
for k in feature_dict:
    f.write(f"{k} : {feature_dict[k]}\n")
f.close()
