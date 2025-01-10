from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from euroleague_api.game_stats import GameStats
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from elfantasy.config import DATA_DIR
from elfantasy.functions import (
    calculate_game_codes,
    calculate_running_standings,
    calculate_standings,
    get_euroleague_data,
    optimize_team,
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

standings_running_features = [
    "Round",
    "TeamCode",
    "HomeWinRate",
    "AwayWinRate",
    "WinRate",
    "WinsLast1Games",
    "WinsLast3Games",
    "WinsLast5Games",
]

home_standings = (
    standings_running[standings_running_features]
    .assign(Round=lambda x: x.Round + 1)
    .rename(columns={c: f"HomeTeam_{c}" for c in standings_running_features if c not in ["Round", "TeamCode"]})
)

away_standings = (
    standings_running[standings_running_features]
    .assign(Round=lambda x: x.Round + 1)
    .rename(columns={c: f"AwayTeam_{c}" for c in standings_running_features if c not in ["Round", "TeamCode"]})
)

df_team_features = (
    df.merge(home_standings, left_on=["week", "hometeamcode"], right_on=["Round", "TeamCode"], how="left")
    .drop(columns=["Round", "TeamCode"])
    .merge(away_standings, left_on=["week", "awayteamcode"], right_on=["Round", "TeamCode"], how="left")
    .drop(columns=["Round", "TeamCode"])
)

df_predict = df_team_features.copy()

# if were to use the previous week's valuation as a predictor
df_predict["valuation_lag_1"] = (
    df_predict.sort_values(by=["week"], ascending=True).groupby(["slug"])["valuation"].shift()
)

# if were to use the rolling average of the previous 3 weeks as a predictor
df_predict["valuation_roll_3"] = (
    df_predict.sort_values(by=["week"], ascending=True)
    .groupby(["slug"])["valuation"]
    .rolling(3, closed="left")
    .mean()
    .reset_index(0, drop=True)
)

# Calculate errors
df_predict["abs_error_lag_1"] = abs(df_predict["valuation"] - df_predict["valuation_lag_1"])
df_predict["abs_error_roll_3"] = abs(df_predict["valuation"] - df_predict["valuation_roll_3"])

# Review
df_predict[["abs_error_lag_1", "abs_error_roll_3"]].describe()

"""
# ==============================================================
# Model Training
# ==============================================================
"""

model_columns = [
    "slug",
    "position_id",
    "team_code",
    "home_away",
    "HomeTeam_HomeWinRate",
    "HomeTeam_AwayWinRate",
    "HomeTeam_WinRate",
    "HomeTeam_WinsLast1Games",
    "HomeTeam_WinsLast3Games",
    "HomeTeam_WinsLast5Games",
    "AwayTeam_HomeWinRate",
    "AwayTeam_AwayWinRate",
    "AwayTeam_WinRate",
    "AwayTeam_WinsLast1Games",
    "AwayTeam_WinsLast3Games",
    "AwayTeam_WinsLast5Games",
    "valuation_lag_1",
    "valuation_roll_3",
    "valuation",
]

# Define feature columns
categorical_features = ["slug", "team_code"]

categorical_features = ["team_code"]
one_hot_features = ["home_away", "position_id"]
numerical_features = [
    "HomeTeam_HomeWinRate",
    "HomeTeam_AwayWinRate",
    "HomeTeam_WinRate",
    "HomeTeam_WinsLast1Games",
    "HomeTeam_WinsLast3Games",
    "HomeTeam_WinsLast5Games",
    "AwayTeam_HomeWinRate",
    "AwayTeam_AwayWinRate",
    "AwayTeam_WinRate",
    "AwayTeam_WinsLast1Games",
    "AwayTeam_WinsLast3Games",
    "AwayTeam_WinsLast5Games",
    "valuation_lag_1",
    "valuation_roll_3",
]

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = TargetEncoder()
one_hot_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("onehot", one_hot_transformer, one_hot_features),
    ]
)


# Define the models
model_lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=1990,
)

model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=1990,
)

model_knn = KNeighborsRegressor(
    n_neighbors=10,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
)

model = model_lgbm

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into train and test sets
modeling_data = df_predict[model_columns].dropna()

X = modeling_data[model_columns].drop(columns=["valuation"])
y = modeling_data["valuation"]

# preprocessor.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1990, shuffle=False)

[x.shape for x in [X_train, X_test, y_train, y_test]]

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# Extract feature importances
feature_importances = pipeline.named_steps["model"].feature_importances_
feature_names = (
    numerical_features
    + categorical_features
    + list(pipeline.named_steps["preprocessor"].named_transformers_["onehot"].get_feature_names_out(one_hot_features))
)

# Create a DataFrame for plotting
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(
    by="Importance", ascending=False
)

importance_df.plot(kind="barh", x="Feature", y="Importance", legend=False, figsize=(10, 8))


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
