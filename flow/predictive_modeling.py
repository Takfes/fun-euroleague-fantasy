import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from elfantasy.config import Configuration
from elfantasy.modeling import build_pred_model
from elfantasy.utils import get_timetag, read_datalog, update_datalog

# configuration
config = Configuration(use_dotenv_config_yaml=True)
data_dir_features_data = config.data_dir_features_data
data_dir_predictions = config.data_dir_predictions
datalog = read_datalog()
training_min_weeks = config.training_min_weeks
baseline_column_predictions = config.baseline_column_predictions
model_column_predictions = config.model_column_predictions
predictive_model_string = config.predictive_model_string
predictive_model_id_cols = config.predictive_model_id_cols
predictive_model_cats_cols = config.predictive_model_cats_cols

# timetag
timetag = get_timetag()

# read features data
euroleague_features_path = data_dir_features_data / datalog["features"]
df_features = pd.read_csv(euroleague_features_path)

# Define the data
idcols = predictive_model_id_cols
# Define the features
feats = [x for x in df_features.columns if ("_lnp_sttc" in x) or ("_plr_sttc" in x) or ("_plr_tmpr" in x)]
# Define feature columns
categorical_features = predictive_model_cats_cols
numerical_features = [x for x in feats if x not in categorical_features]
# Define the target
target = "valuation"
# Define the model columns
model_columns = idcols + feats + [target]

# Define the design matrix
design_matrix = df_features[model_columns].copy()

# Split data into train and test sets
X = design_matrix.drop(columns=[target])
y = design_matrix[target]

# Make the estimator
estimator = build_pred_model(predictive_model_string, numerical_features, categorical_features, transform_target=True)

# Setup model training params
store_predictions_for_first_train = False
verbose = True
total_weeks = design_matrix["week"].nunique()
df_predictions = df_features.copy()
maes_baseline = []
maes = []

for index, current_week in enumerate(range(training_min_weeks, total_weeks), start=1):
    train_indices = df_features[df_features["week"] <= current_week].index
    test_indices = df_features[df_features["week"] == current_week + 1].index

    X_train = X.loc[train_indices].drop(columns=["week"])
    y_train = y.loc[train_indices]
    X_test = X.loc[test_indices].drop(columns=["week"])
    y_test = y.loc[test_indices]

    # Fit the model
    estimator.fit(X_train, y_train)

    # Make predictions
    y_pred = estimator.predict(X_test)

    # Store predictions in the df_predictions dataframe
    if index == 1 and store_predictions_for_first_train:
        df_predictions.loc[train_indices, model_column_predictions] = estimator.predict(X_train)

    df_predictions.loc[test_indices, model_column_predictions] = y_pred

    # Calculate the MAE
    y_pred_baseline = df_predictions.loc[test_indices][baseline_column_predictions]
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mae = mean_absolute_error(y_test, y_pred)
    maes_baseline.append(mae_baseline.item())
    maes.append(mae.item())

    # Print the results
    if verbose:
        print(f"Week {current_week + 1}) MAE  : {mae:.2f}, Baseline : {mae_baseline:.2f}")
        print(f"- Improvement : {mae_baseline - mae:.2f}, Model is Better : {mae_baseline > mae}")
        print(f"- Running Baseline: {np.mean(maes_baseline):.2f} +/- {np.std(maes_baseline):.2f}")
        print(f"- Running MAE :     {np.mean(maes):.2f} +/- {np.std(maes):.2f}")
        print()


print(f"Final Results {predictive_model_string} model")
print(f"Baseline column {baseline_column_predictions}")
print(f"- Running Baseline: {np.mean(maes_baseline):.2f} +/- {np.std(maes_baseline):.2f}")
print(f"- Running MAE :     {np.mean(maes):.2f} +/- {np.std(maes):.2f}")


# save predictions
predictions_file_name = f"predictions_{timetag}.csv"
df_predictions.to_csv(data_dir_predictions / predictions_file_name, index=False)

# update datalog
datalog_update = {"predictions": predictions_file_name}
update_datalog(datalog_update)
