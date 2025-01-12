import pandas as pd

from elfantasy.config import Configuration
from elfantasy.optimization import build_opt_model
from elfantasy.utils import get_timetag, read_datalog, update_datalog

# configuration
config = Configuration()
data_dir_predictions = config.data_dir_predictions
data_dir_optimizations = config.data_dir_optimizations
training_min_weeks = config.training_min_weeks
baseline_column = config.baseline_column
prediction_column = config.prediction_column
datalog = read_datalog()

# timetag
timetag = get_timetag()

# read features data
euroleague_features_path = data_dir_predictions / datalog["predictions"]
df_predictions = pd.read_csv(euroleague_features_path)

# Initialize objects to store solutions
solutions_optimal = {}
solutions_baseline = {}
solutions_model = {}

# Optimize teams for each week using different predictors
for w in range(training_min_weeks + 1, df_predictions.week.max().item() + 1):
    print(f"Optimizing week {w}")
    dfw = df_predictions[df_predictions["week"] == w].reset_index(drop=True).copy()
    # Optimal
    sol = build_opt_model(dfw, value_col="valuation", budget=100)
    solutions_optimal[w] = sol
    # Baseline
    sol = build_opt_model(dfw, value_col=baseline_column, budget=100)
    solutions_baseline[w] = sol
    # Model
    sol = build_opt_model(dfw, value_col=prediction_column, budget=100)
    solutions_model[w] = sol

# Collect results

df_optimizations = pd.concat(
    [
        pd.DataFrame(solutions_optimal).T.rename(columns=lambda x: f"{x}_optimal"),
        pd.DataFrame(solutions_baseline).T.rename(columns=lambda x: f"{x}_baseline"),
        pd.DataFrame(solutions_model).T.rename(columns=lambda x: f"{x}_model"),
    ],
    axis=1,
)

# Calculate differences
df_optimizations["obj_opt_vs_baseline"] = (
    df_optimizations["objective_value_optimal"] - df_optimizations["objective_value_baseline"]
).astype(float)
df_optimizations["obj_opt_vs_model"] = (
    df_optimizations["objective_value_optimal"] - df_optimizations["objective_value_model"]
).astype(float)

# save optimizations
optimizations_file_name = f"optimizations_{timetag}.csv"
df_optimizations.to_csv(data_dir_optimizations / optimizations_file_name, index=True)


# update datalog
datalog_update = {"optimizations": optimizations_file_name}
update_datalog(datalog_update)
