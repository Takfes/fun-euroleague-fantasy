from pathlib import Path

import numpy as np
import pandas as pd
from euroleague_api.game_stats import GameStats

from elfantasy.config import DATA_DIR
from elfantasy.functions import (
    calculate_running_standings_from_games,
    calculate_standings_from_games,
    get_euroleague_data,
    optimize_team,
    plot_stats_boxes,
    plot_stats_lines,
    tidy_euroleague_data,
    tidy_games_data,
)

pd.set_option("display.max_columns", None)
competition_code = "E"
season = 2024

# gamestats
gamestats = GameStats(competition_code)
games_raw = gamestats.get_game_reports_single_season(season)
games = tidy_games_data(games_raw)
standings = calculate_standings_from_games(games)
standings_running = calculate_running_standings_from_games(games)

# euroleague data
df_raw = get_euroleague_data()
df = tidy_euroleague_data(df_raw)

"""
# ==============================================================
# Simulation - Predictions
# ==============================================================
"""

# bring opponent data in the dataframe

games[
    [
        "Round",
        "Gamecode",
        "HomeTeamCode",
        "AwayTeamCode",
    ]
].to_records(index=False)

# TODO add features to reflect next match difficulty
# TODO add features to reflect home/away advantage
# TODO add features to reflect home/away strength
# TODO add features to reflect team form
# TODO add features to reflect player form

"""
# ==============================================================
# Simulation - Predictions
# ==============================================================
"""

# if were to use the previous week's valuation as a predictor
df_predict = df.copy()
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
df_predict["error_lag_1"] = df_predict["valuation"] - df_predict["valuation_lag_1"]
df_predict["error_roll_3"] = df_predict["valuation"] - df_predict["valuation_roll_3"]

# Review
df_predict[["error_lag_1", "error_roll_3"]].describe()

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
