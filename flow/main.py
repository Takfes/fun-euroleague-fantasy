from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from euroleague_api.game_stats import GameStats
from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, PowerTransformer, StandardScaler
from xgboost import XGBRegressor

from elfantasy.config import DATA_DIR
from elfantasy.functions import (
    calculate_game_codes,
    calculate_running_standings,
    calculate_standings,
    get_euroleague_data,
    make_player_contribution,
    make_player_rolling_stats,
    make_team_form,
    model_make_estimator,
    model_make_hpo_estimator,
    optimize_team,
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

df1 = make_team_form(df, standings_running)
df2 = make_player_contribution(df1)
df3 = make_player_rolling_stats(df2)

"""
# ==============================================================
# Model Training
# ==============================================================
"""


# Define baselines
baseline_cols = [
    "valuation_lag_1_mprs",
    "valuation_lag_3_mprs",
    "valuation_roll_3_mprs",
]

baseline_results = {}
for bc in baseline_cols:
    baseline_results[bc] = round(mean_absolute_error(df3["valuation"], df3[bc]).item(), 4)

print(baseline_results)

# Define the data
idcols = [
    "week",
    "slug",
    "team_code",
]
# Define the features
feats = [x for x in df3.columns if ("mtf" in x) or ("mpc" in x) or ("mprs" in x)]
# Define feature columns
categorical_features = ["team_code", "slug"]
numerical_features = [x for x in feats if x not in categorical_features]
# Define the target
target = "valuation"
# Define the model columns
model_columns = idcols + feats + [target]
# Define the design matrix
design_matrix = df3[model_columns].sort_values(by="week", ascending=True).dropna().reset_index(drop=True)

# Split data into train and test sets
X = design_matrix[model_columns].drop(columns=["valuation"])
y = design_matrix["valuation"]

# Make the estimator
model_string = "XGB"
estimator = model_make_estimator(model_string, numerical_features, categorical_features)

# Parameters
min_weeks = 3  # Minimum number of weeks for initial training
total_weeks = design_matrix["week"].nunique()

# Cross-validation
maes = []

for current_week in range(min_weeks, total_weeks):
    train_indices = X[X["week"] <= current_week].index
    test_indices = X[X["week"] == current_week + 1].index

    X_train = X.loc[train_indices].drop(columns=["week"])
    y_train = y.loc[train_indices]
    X_test = X.loc[test_indices].drop(columns=["week"])
    y_test = y.loc[test_indices]

    pt = PowerTransformer(method="yeo-johnson")
    # Fit and transform the training target variable
    y_train_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    # Transform the test target variable
    y_test_transformed = pt.transform(y_test.values.reshape(-1, 1)).flatten()

    # Fit the model
    estimator.fit(X_train, y_train_transformed)

    # Make predictions
    y_pred_transformed = estimator.predict(X_test)
    y_pred = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae.item())
    print(f"Mean Absolute Error: for {current_week=} {mae:.2f}")
    print(f"Mean Absolute Error: {mae:.2f} +/- {np.std(maes):.2f}")
    print()

# Call the function with the appropriate arguments
plot_regression_diagnostics(y_test, y_pred)


"""
# ==============================================================
# Simulation - Predictions
# ==============================================================
"""

# Initialize objects to store solutions
solutions_optimal = {}
solutions_lag_1 = {}
solutions_roll_3 = {}

# Optimize teams for each week using different predictors
for w in range(4, df_predict.week.max().item() + 1):
    print(f"Optimizing week {w}")
    dfw = df_predict[df_predict["week"] == w].reset_index(drop=True).copy()
    # Optimal
    team, obj, budget = optimize_team(dfw, value_col="valuation", budget=100)
    solutions_optimal[w] = {"team": team, "obj": obj, "budget": budget}
    # Lag 1
    team, obj, budget = optimize_team(dfw, value_col="valuation_lag_1", budget=100)
    solutions_lag_1[w] = {"team": team, "obj": obj, "budget": budget}
    # Roll 3
    team, obj, budget = optimize_team(dfw, value_col="valuation_roll_3", budget=100)
    solutions_roll_3[w] = {"team": team, "obj": obj, "budget": budget}

# Collect results
simres = pd.concat(
    [
        pd.DataFrame(solutions_optimal).T.rename(columns=lambda x: f"{x}_optimal"),
        pd.DataFrame(solutions_lag_1).T.rename(columns=lambda x: f"{x}_lag_1"),
        pd.DataFrame(solutions_roll_3).T.rename(columns=lambda x: f"{x}_roll_3"),
    ],
    axis=1,
)[
    [
        "obj_optimal",
        "obj_lag_1",
        "obj_roll_3",
        "budget_lag_1",
        "budget_optimal",
        "budget_roll_3",
        "team_optimal",
        "team_lag_1",
        "team_roll_3",
    ]
]

# Calculate differences
simres["obj_opt_vs_lag_1"] = (simres["obj_optimal"] - simres["obj_lag_1"]).astype(float)
simres["obj_opt_vs_roll_3"] = (simres["obj_optimal"] - simres["obj_roll_3"]).astype(float)

# Review
simres[["obj_opt_vs_lag_1", "obj_opt_vs_roll_3"]].describe()

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
