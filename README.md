
## Kaggle - American Express Default Prediction
The kaggle competition: https://www.kaggle.com/competitions/amex-default-prediction

There are files in this repository which were not included due to their size (e.g. AMEX provided csv files containing training and test datasets). To run files in this repo (and thus re-produce results in the report), do the following:
1. Download the train_data.csv, test_data.csv and train_labels.csv from the kaggle competition and save it in the data_preprocessing directory
2. Run the reformat_to_parquet.py file to convert csv files to parquet files
3. To obtain Feature Set 1 and Feature Set 2 run create_feature_set_1.py and create_feature_set_2.py

The necessary files will now be obtained to run the scripts contained within the other directories to reproduce results.

## Directory Map
| Folder | Description |
| ------ | ----------- |
| data_preprocessing | Data preprocessing scripts |
| feature_engineering | Feature Engineering scripts to analyse whether newly created features should be added to Feature Set 2 |
| optimisation | Feature Selection & Parameter Tuning scripts |
| kaggle_submissions | Scripts to build kaggle submissions, screenshots kaggle submissions and cross validated scores. Uses default, optimized models and different feature sets |
| results | Image or csv results of optimisation |
| stacking | Model stacking scripts to stack the different models |
| helper_functions | Utility scripts |

The models used are XGBoost, LightGBM, sklearn Random Forest Classifier and cuML Logistic Regression.

## Data Preprocessing
Scripts for data preprocessing is located at the "data_preprocessing" folder.
Feature set 1 is uses base features provided by AMEX with OHE applied on categorial features; only the customer's final statement is extracted.
Feature set 2 is the base features with added engineered features.
| Script | Description | Creates |
| ------ | ----------- | ------- |
| reformat_to_parquet.py | Perform data preprocessing and exports the csv into parquet | test_data.parquet, train_data.parquet |
| create_feature_set_1.py | Creates a test and train dataset of feature set 1 | X_test_feature_set_1.parquet, X_train_feature_set_1.parquet |
| create_feature_set_2.py | Creates a test and train dataset of feature set 2 (Added engineered features) | X_test_feature_set_2.parquet, X_train_feature_set_2.parquet |

Firstly, create the parquet file from Kaggle's csv data file using reformat_to_parquet. Create the different feature sets using create_feature_set_1.py and create_feature_set_2.py. Feature_Set_Features.txt also contained in this directory, it provides the features which were contained within Feature Set 1 and 2.

## Feature Engineering
Scripts for analysis of newly created features is located in the "feature_engineering" folder.
Also contains results, correlation_features.txt, of analysis described in Section 4.2 of final report.
| Script | Description | Creates |
| ------ | ----------- | ------- |
| features_corr.py | Perform analysis using Pearson Correlation on newly created features to determine whether they should added to Feature Set 2 | correlation_features.txt |


## Model Optimisation
Scripts for model optimization can be found in the "optimisation" folder.
Inside there are separate folders containing different models.
### XGBoost
| Script | Description | Creates |
| ------ | ----------- | ------- |
| rfecv_feature_set_2.py | RFECV for feature selection | XGB_cv_scores.png, XGB_feature_ranking.csv |
| optuna_hyperparameter_tuning.py | Optuna parameter tuning script | XGB_optimization_history_optimal_features.png, XGB_parallel_coordinate_optimal_features.png, XGB_param_importances_optimal_features.png, XGB_study.pkl |

### LightGBM
| Script | Description | Creates |
| ------ | ----------- | ------- |
| rfecv_feature_set_2.py | RFECV for feature selection | LightGBM_cv_scores.png, LightGBM_feature_ranking.csv |
| optuna_hyperparameter_tuning.py | Optuna parameter tuning script | LightGBM_optimization_history_optimal_features.png, LightGBM_parallel_coordinate_optimal_features.png, LightGBM_param_importances_optimal_features.png, LightGBM_study.pkl |


### Logistic Regression
cuML's logistic regression is used.
| Script | Description | Creates |
| ------ | ----------- | ------- |
| feature_selection.py | Uses L1 to do feature selection and exports pickle files of each cross validation folds | cv_results_{i} (Used to create L1_eliminated_features.pkl) |
| feature_selection_extract.py | Extracts the average accuracy and average features eliminated from the cross validation folds file | |
| feature_selection_plots.py | Plots the accuracy and features eliminated of every cross validation fold | FeatureSelection_NestedCV.png, FeatureSelection_AvgFold.png |
| feature_selection_process_folds.py | Selects the eliminated features based on the averaged cross validation fold | L1_eliminated_features.pkl |
| optuna_hyperparameter_tuning.py | Optuna parameter tuning script | logreg_param_importances.png, logreg_optimization_history.png, logreg_parallel_coordinate.png, study |
| feature_importance.py | Creates the coef importance plot | LogisticRegression_FeatureImportance.png |

Run the feature_selection.py to produce all the cross validation fold information of feature selected through L1, it will all be in cv_results directory. Extract all the averages of accuracy and feature eliminated with feature_selection_extract.py. Plot it with the feature_selection_plot.py. To get the eliminated features of L1, run feature_selection_process_folds.py. After getting the features, optimize the parameter with Optuna by running hyperparameter_tuning_optimal_features.py with the extracted features. Finally, we have the optimized logistic regression model.

### Random Forest Classifier
| Script | Description | Creates |
| ------ | ----------- | ------- |
| feature_selection.py | RFECV for feature selection | selected_features, df_selected_features (Used to create selected_features.csv) |
| optuna_hyperparameter_tuning.py | Optuna parameter tuning script | rf_param_importances.png, rf_optimization_history.png, rf_parallel_coordinate.png, study |

## Kaggle Submissions
Provides screenshots of the results given by Kaggle from the submissions of the respective model. Also provides the cross validated score in a .txt file.
| Script | Description | Creates |
| ------ | ----------- | ------- |
| generate_kaggle_submission.py | Generates 3 Kaggle submissions using the respective model | submission_{modelpostfix}_{i}.csv |

3 generated submissions are for: 
1. Feature Set 1, Default Hyperparameters
2. Optimal Features, Default Hyperparameters
3. Optimal Features, Optimal Hyperparameters

e.g. for 3. the submission file for LGBMClassifier will be submission_LGBM_3.csv

## Results
In each respective directory there is the output of
- Feature Selection ({model}_cv_scores.png, {model}_feature_ranking.csv)
- Hyperparameter Optimisation ({model}_optimisation_history_optimal_features.png, {model}_parallel_coordinate_optimal_features.png, {model}_param_importances_optimal_features.png)

A pickle file ({model}_study.pkl) is also provided which can be used to reproduce the plots in Hyperparameter Optimisation from the Optuna optimisation

Logistic Regression - Additional Result Files:
| File | Description |
| ---- | ----------- |
| logreg_feature_selection.csv | The feature selected |
| FeatureSelection_NestedCV.png | The 5 nested cross validation folds plot for feature selection |
| FeatureSelection_AvgFold.png | The averaged nested cross validation folds plot for feature selection |
| LogisticRegression_FeatureImportance.png | The feature importance plot using the coef_ as importance score |

## Stacking
Each model has their own folder that contains a script to generate its individual predictions used for stacking, the scripts are named {model}_stack.py.

The output of {model}_stack.py
| File | Description |
| ---- | ----------- |
| stacking_train_pred.csv | The train predictions for each models |
| stacking_Y_train.csv | The correct Y label for the train dataset |
| stacking_test_pred.csv | The 5 folds test prediction for each models |

Stacking script
| Script | Description | Creates |
| ------ | ----------- | ------- |
| stack.py | Produces train and test predictions for all the 4 models. Tunes and creates the final stacking model. Generate kaggle submission | param_importances.png, optimization_history.png, parallel_coordinate.png, submission_stacked_model.csv
| stack_train.py | Function to train the stacking model using 5 fold cross validation | |
