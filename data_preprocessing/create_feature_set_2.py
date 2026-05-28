import pandas as pd
import numpy as np
import gc

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

# Features Engineered - To be added to the new dataset
engineered_features = [
    "D_39_6_periods_max",
    "B_2_3_periods_logmean",
    "R_1_13_periods_logmean",
    "S_3_13_periods_max",
    "D_41_3_periods_logmax",
    "B_3_3_periods_logmax",
    "D_42_13_periods_max",
    "D_43_13_periods_max",
    "D_44_6_periods_logmean",
    "D_45_13_periods_logmedian",
    "B_5_6_periods_logmax",
    "R_2_6_periods_max",
    "D_46_13_periods_mean",
    "D_47_13_periods_min",
    "D_48_3_periods_mean",
    "D_49_13_periods_logmin",
    "B_6_13_periods_min",
    "B_8_13_periods_min",
    "D_50_13_periods_sum",
    "D_51_13_periods_logsum",
    "B_9_3_periods_min",
    "R_3_13_periods_mean",
    "D_52_13_periods_min",
    "P_3_6_periods_min",
    "B_10_6_periods_min",
    "D_53_6_periods_std",
    "S_5_13_periods_logmedian",
    "B_11_3_periods_logmean",
    "S_6_13_periods_std",
    "R_4_13_periods_std",
    "S_7_13_periods_max",
    "B_12_13_periods_logsum",
    "S_8_6_periods_min",
    "D_56_13_periods_sum",
    "B_13_13_periods_loglast/first",
    "R_5_13_periods_logmean",
    "S_9_13_periods_logmean",
    "B_14_6_periods_min",
    "D_59_3_periods_mean",
    "D_60_6_periods_logmin",
    "D_61_6_periods_mean",
    "B_15_3_periods_std",
    "S_11_13_periods_min",
    "D_62_13_periods_mean",
    "D_65_6_periods_logmax",
    "B_17_6_periods_max",
    "B_19_3_periods_logmean",
    "B_20_3_periods_logmean",
    "S_12_13_periods_min",
    "R_6_13_periods_logmax",
    "S_13_6_periods_min",
    "B_21_13_periods_logmean",
    "D_69_13_periods_logmax",
    "B_22_6_periods_logmean",
    "D_70_13_periods_logmean",
    "D_71_3_periods_logmin",
    "D_72_13_periods_logmax",
    "S_15_13_periods_max",
    "B_23_3_periods_logmean",
    "D_73_13_periods_sum",
    "P_4_13_periods_first",
    "D_74_3_periods_logmean",
    "D_76_13_periods_sum",
    "B_24_6_periods_logmax",
    "R_7_6_periods_logmax",
    "D_77_13_periods_logmin",
    "B_25_3_periods_min",
    "B_26_3_periods_logmax",
    "D_78_13_periods_logmax",
    "D_79_13_periods_logmax",
    "R_8_13_periods_logmean",
    "R_9_13_periods_std",
    "S_16_13_periods_logmean",
    "D_80_13_periods_std",
    "R_10_13_periods_logmean",
    "R_11_13_periods_mean",
    "B_27_13_periods_sum",
    "D_81_13_periods_logmax",
    "D_82_13_periods_sum",
    "S_17_13_periods_std",
    "R_12_6_periods_min",
    "R_13_13_periods_logmax",
    "D_83_13_periods_logmax",
    "R_14_13_periods_logmax",
    "R_15_13_periods_std",
    "D_84_13_periods_logmax",
    "R_16_13_periods_std",
    "B_29_13_periods_loglast/first",
    "S_18_13_periods_logsum",
    "D_86_13_periods_logsum",
    "D_87_13_periods_sum",
    "R_17_13_periods_logmax",
    "R_18_13_periods_sum",
    "D_88_13_periods_sum",
    "B_31_13_periods_std",
    "S_19_13_periods_logsum",
    "B_32_13_periods_max",
    "S_20_13_periods_max",
    "R_20_13_periods_logmax",
    "R_21_13_periods_max",
    "B_33_3_periods_sum",
    "D_89_13_periods_logmax",
    "R_22_13_periods_max",
    "R_23_13_periods_logsum",
    "D_91_13_periods_max",
    "D_92_13_periods_logsum",
    "D_93_13_periods_logsum",
    "D_94_13_periods_logsum",
    "R_24_13_periods_max",
    "R_25_13_periods_std",
    "D_96_13_periods_logsum",
    "S_22_3_periods_max",
    "S_23_3_periods_median",
    "S_24_3_periods_max",
    "S_25_13_periods_std",
    "S_26_13_periods_logsum",
    "D_102_13_periods_logmax",
    "D_103_13_periods_logmax",
    "D_104_13_periods_logmax",
    "D_105_13_periods_sum",
    "D_106_13_periods_logmin",
    "D_107_13_periods_logmax",
    "B_36_6_periods_mean",
    "R_26_3_periods_std",
    "R_27_13_periods_sum",
    "D_108_13_periods_sum",
    "D_109_13_periods_logsum",
    "D_110_13_periods_sum",
    "D_111_13_periods_sum",
    "B_39_13_periods_sum",
    "D_112_1_periods_logfirst",
    "B_40_3_periods_logmean",
    "S_27_13_periods_logmin",
    "D_113_13_periods_logmax",
    "D_115_13_periods_sum",
    "D_118_13_periods_sum",
    "D_119_13_periods_sum",
    "D_121_13_periods_sum",
    "D_122_13_periods_sum",
    "D_123_13_periods_logmax",
    "D_124_13_periods_std",
    "D_125_13_periods_logmax",
    "D_127_13_periods_logsum",
    "D_128_13_periods_min",
    "D_129_13_periods_sum",
    "B_41_6_periods_logmax",
    "B_42_13_periods_logsum",
    "D_130_13_periods_logmax",
    "D_131_13_periods_logmax",
    "D_132_13_periods_std",
    "D_133_13_periods_logmax",
    "R_28_13_periods_logsum",
    "D_134_13_periods_std",
    "D_135_13_periods_logmin",
    "D_136_13_periods_std",
    "D_137_13_periods_logmin",
    "D_138_13_periods_std",
    "D_139_13_periods_logmax",
    "D_140_13_periods_logmax",
    "D_141_13_periods_logmax",
    "D_142_13_periods_sum",
    "D_143_13_periods_logmax",
    "D_144_13_periods_logfirst",
    "D_145_13_periods_logmax",
    "B_30_0.0_6_periods_min",
    "B_30_1.0_6_periods_max",
    "B_30_2.0_13_periods_max",
    "B_38_2.0_13_periods_mean",
    "B_38_3.0_13_periods_std",
    "B_38_4.0_13_periods_std",
    "B_38_5.0_13_periods_std",
    "B_38_6.0_13_periods_max",
    "B_38_7.0_13_periods_max",
    "D_114_0.0_6_periods_max",
    "D_114_1.0_13_periods_first",
    "D_116_0.0_13_periods_min",
    "D_116_1.0_13_periods_max",
    "D_117_-1.0_6_periods_std",
    "D_117_1.0_13_periods_max",
    "D_117_2.0_13_periods_max",
    "D_117_3.0_13_periods_std",
    "D_117_4.0_13_periods_sum",
    "D_117_5.0_13_periods_max",
    "D_117_6.0_13_periods_sum",
    "D_120_0.0_13_periods_min",
    "D_120_1.0_13_periods_max",
    "D_126_-1.0_13_periods_last-first",
    "D_126_0.0_13_periods_median",
    "D_126_1.0_13_periods_first",
    "D_63_CO_13_periods_first",
    "D_63_CR_13_periods_sum",
    "D_63_XL_3_periods_max",
    "D_63_XM_13_periods_std",
    "D_63_XZ_13_periods_std",
    "D_64_-1_13_periods_mean",
    "D_64_O_13_periods_sum",
    "D_64_R_13_periods_max",
    "D_64_U_6_periods_max",
    "D_66_0.0_13_periods_std",
    "D_66_1.0_13_periods_min",
    "D_68_0.0_13_periods_mean",
    "D_68_1.0_13_periods_std",
    "D_68_2.0_13_periods_std",
    "D_68_3.0_13_periods_max",
    "D_68_4.0_13_periods_max",
    "D_68_5.0_13_periods_std",
    "D_68_6.0_13_periods_first",
]

# Categorical Features
categorical_features = {
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
}

# Files to update (Train and Test)
files = [("train", config["DATAPATH"]["Parquet_Train_Path"], config["DATAPATH"]["Train_FeatureSet2_Path"]), 
        ("test", config["DATAPATH"]["Parquet_Test_Path"], config["DATAPATH"]["Test_FeatureSet2_Path"])]

for file, file_path, processed_file_path in files:
    print(f"Reading in {file_path} file")
    data = pd.read_parquet(file_path)
    print(f"{file_path} successfully imported")

    # Group by customer_ID
    data_for_model = data.groupby("customer_ID").last().sort_index()
    data_for_model.fillna((data_for_model.mean()), inplace=True)

    # Create engineered features.
    for i in range(len(engineered_features)):

        print(f"Including feature: {engineered_features[i]}")

        # Method to apply to the feature and period of record aggregation.
        feature = engineered_features[i].split("_")
        method = feature[-1]
        period = int(feature[-3])
        feature = (
            f"{feature[0]}_{feature[1]}"
            if f"{feature[0]}_{feature[1]}" not in categorical_features
            else f"{feature[0]}_{feature[1]}_{feature[2]}"
        )

        # Aggregate customer's data through a period of time.
        cols_array = ["customer_ID"] + [feature]
        period_last_entries_for_customers = data.groupby(["customer_ID"])[
            cols_array
        ].tail(period)

        nulls_in_data = period_last_entries_for_customers[feature].isnull().values.any()

        if nulls_in_data:
            period_last_entries_for_customers[feature].fillna(
                (period_last_entries_for_customers[feature].mean()), inplace=True
            )

        log_feature = False
        if "log" in method:
            log_feature = True
            method = method[3:]

        if method == "std":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).std()
        elif method == "median":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).median()
        elif method == "min":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).min()
        elif method == "max":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).max()
        elif method == "sum":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).sum()
        elif method == "first":
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).first()
        elif method == "last-first":
            first = period_last_entries_for_customers.groupby(["customer_ID"]).first()
            last = period_last_entries_for_customers.groupby(["customer_ID"]).last()
            feature_to_add = last.subtract(first)
        elif method == "last/first":
            first = period_last_entries_for_customers.groupby(["customer_ID"]).first()
            last = period_last_entries_for_customers.groupby(["customer_ID"]).last()
            feature_to_add = last.divide(first)
        else:
            feature_to_add = period_last_entries_for_customers.groupby(
                ["customer_ID"]
            ).mean()

        if log_feature and (feature_to_add[feature] > 0).all():
            feature_to_add[feature] = np.log2(feature_to_add[feature])
        elif log_feature:
            continue

        new_feature_name = engineered_features[i]
        feature_to_add.rename(columns={feature: new_feature_name}, inplace=True)

        data_for_model = data_for_model.join(feature_to_add[new_feature_name])
        print(f"{new_feature_name} added to {processed_file_path}")

    # Export dataset with engineered features to parquet.
    del feature_to_add, period_last_entries_for_customers
    data_for_model.to_parquet(processed_file_path)
    del data_for_model, data
    gc.collect()


print("Dataset with engineered features created.")
