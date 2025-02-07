from pathlib import Path

# import highspy
import numpy as np
import pandas as pd
from euroleague_api.game_stats import GameStats
from sklearn.metrics import mean_absolute_error

from elfantasy.config import DATA_DIR
from elfantasy.functions import (
    build_opt_model,
    build_pred_model,
    calculate_game_codes,
    calculate_running_standings,
    calculate_standings,
    get_euroleague_data,
    make_lineup_static_feats,
    make_player_static_feats,
    make_player_tempor_feats,
    plot_regression_diagnostics,
    plot_stats_boxes,
    plot_stats_lines,
    tidy_euroleague_data,
    tidy_games_data,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

competition_code = "E"
season = 2024

# gamestats
gamestats = GameStats(competition_code)
games_raw = gamestats.get_game_reports_single_season(season)
games_raw_static = games_raw.loc[games_raw["Round"] <= 19]
games = tidy_games_data(games_raw_static)
game_codes = calculate_game_codes(games)

# standings
standings = calculate_standings(games)
standings_running = calculate_running_standings(games, game_codes)

# euroleague data
df_raw = get_euroleague_data()
df_raw_static = df_raw.loc[df_raw["week"] <= 19]
df = tidy_euroleague_data(df_raw_static, games, game_codes)

"""
# ==============================================================
# Feature Engineering
# ==============================================================
"""

# TODO : add all time valuation and its std per player and week
# TODO : create feature class that will work per week - valuation of all available observations vs relying on preconstructed features - create features on demand
# TODO : explore features through correlation matrix - then create interactions
# TODO : explore GroupedCrossValidation - instead of manual thing

df1 = make_lineup_static_feats(df, standings_running)
df2 = make_player_static_feats(df1)
df3 = make_player_tempor_feats(df2)

df_features = df3.sort_values(by=["week", "team_code", "slug"], ascending=True).reset_index(drop=True).copy()

# corr = df_features.filter(like="mtf").corr()

"""
# ==============================================================
# Model Training
# ==============================================================
"""

# Define the data
idcols = [
    "week",
    "slug",
    "team_code",
]
# Define the features
feats = [x for x in df_features.columns if ("mtf" in x) or ("mpc" in x) or ("mprs" in x)]
# Define feature columns
categorical_features = ["team_code", "slug"]
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
model_string = "LR"
estimator = build_pred_model(model_string, numerical_features, categorical_features, transform_target=True)

# Setup model training params
store_predictions_for_first_train = False
verbose = True
min_weeks = 5  # Minimum number of weeks for initial training
baseline_column = "valuation_roll_3_plr_tmpr"
prediction_column = "valuation_pred"
total_weeks = design_matrix["week"].nunique()
df_predictions = df_features.copy()
maes_baseline = []
maes = []

for index, current_week in enumerate(range(min_weeks, total_weeks), start=1):
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
        df_predictions.loc[train_indices, prediction_column] = estimator.predict(X_train)

    df_predictions.loc[test_indices, prediction_column] = y_pred

    # Calculate the MAE
    y_pred_baseline = df_predictions.loc[test_indices][baseline_column]
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


print(f"Final Results {model_string} model")
print(f"- Running Baseline: {np.mean(maes_baseline):.2f} +/- {np.std(maes_baseline):.2f}")
print(f"- Running MAE :     {np.mean(maes):.2f} +/- {np.std(maes):.2f}")

# Final Results KNN model
# - Running Baseline: 5.99 +/- 0.26
# - Running MAE :     6.13 +/- 0.27

# Final Results LR model
# - Running Baseline: 5.99 +/- 0.26
# - Running MAE :     5.71 +/- 0.29

# Final Results XGB model
# - Running Baseline: 5.99 +/- 0.26
# - Running MAE :     5.82 +/- 0.29

# Final Results LGBM model
# - Running Baseline: 5.99 +/- 0.26
# - Running MAE :     5.79 +/- 0.30

# Review the predictions
review_columns = [
    "slug",
    "first_name",
    "last_name",
    "id",
    "position",
    "position_id",
    "team_code",
    "team_name",
    "team_id",
    "week",
    "game_code",
    "home_away",
    "hometeamcode",
    "awayteamcode",
    "home_away_factor_lnp_sttc",
    "win_diff_lnp_sttc",
    "win_rate_diff_lnp_sttc",
    "win_rate_ha_diff_lnp_sttc",
    "win_last1games_lnp_sttc",
    "win_last3games_lnp_sttc",
    "win_last5games_lnp_sttc",
    "valuation",
    "valuation_roll_3_plr_tmpr",
    prediction_column,
]
rv = df_predictions[review_columns].dropna()
# rv.to_clipboard(index=True)

# Call the function with the appropriate arguments
plot_regression_diagnostics(y_test, y_pred)
plot_regression_diagnostics(rv["valuation"], rv[prediction_column])


"""
# ==============================================================
# Simulation - Predictions
# ==============================================================
"""

# Initialize objects to store solutions
solutions_optimal = {}
solutions_baseline = {}
solutions_model = {}

# Optimize teams for each week using different predictors
for w in range(min_weeks + 1, df_predictions.week.max().item() + 1):
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

simres = pd.concat(
    [
        pd.DataFrame(solutions_optimal).T.rename(columns=lambda x: f"{x}_optimal"),
        pd.DataFrame(solutions_baseline).T.rename(columns=lambda x: f"{x}_baseline"),
        pd.DataFrame(solutions_model).T.rename(columns=lambda x: f"{x}_model"),
    ],
    axis=1,
)

# Calculate differences
simres["obj_opt_vs_baseline"] = (simres["objective_value_optimal"] - simres["objective_value_baseline"]).astype(float)
simres["obj_opt_vs_model"] = (simres["objective_value_optimal"] - simres["objective_value_model"]).astype(float)

# Review
# simres[["obj_opt_vs_baseline", "obj_opt_vs_model"]].describe()
# simres.filter(like="obj")
# simres.to_clipboard(index=True)

"""
# ==============================================================
# Data Exploration
# ==============================================================
"""

plot_stats_boxes(df, position="G", criterion="greedy", category="high", stats_agg_func="std")

plot_stats_lines(df, "theo-maledon")

player_averages = (
    # df.groupby(["slug", "position", "team_name"], as_index=False)
    df.groupby(["position"], as_index=False)
    .agg(
        games=("min", "count"),
        mins_avg=("min", "mean"),
        mins_ttl=("min", "sum"),
        cr=("cr", "mean"),
        valuation_mean=("valuation", "mean"),
        valuation_std=("valuation", "std"),
        plus_minus_avg=("plus_minus", "mean"),
        plus_minus_std=("plus_minus", "std"),
    )
    .assign(value_for_credit=lambda x: x.valuation_mean / x.cr)
    .sort_values("value_for_credit", ascending=False)
    .fillna(0)
)

team_averages = (
    df.groupby(["team_name", "position"], as_index=False)
    .agg(
        cr=("cr", "mean"),
        pts=("pts", "mean"),
        fgm=("fgm", "mean"),
        fga=("fga", "mean"),
        tpm=("tpm", "mean"),
        tpa=("tpa", "mean"),
        ftm=("ftm", "mean"),
        fta=("fta", "mean"),
        ast=("ast", "mean"),
        reb=("reb", "mean"),
        stl=("stl", "mean"),
        blk=("blk", "mean"),
        blka=("blka", "mean"),
        oreb=("oreb", "mean"),
        dreb=("dreb", "mean"),
        tov=("tov", "mean"),
        pf=("pf", "mean"),
        fouls_received=("fouls_received", "mean"),
        plus_minus=("plus_minus", "std"),
        valuation=("valuation", "mean"),
    )
    .assign(fgp=lambda x: x.fgm / x.fga, tpp=lambda x: x.tpm / x.tpa, ftp=lambda x: x.ftm / x.fta)
    .drop(columns=["fgm", "fga", "tpm", "tpa", "ftm", "fta"])
)

# Review
games.to_clipboard(index=False)
df.to_clipboard(index=False)
player_averages.to_clipboard(index=False)
team_averages.to_clipboard(index=False)
