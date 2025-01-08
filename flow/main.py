from itertools import count
from operator import index
from pathlib import Path

import pandas as pd
from euroleague_api.game_stats import GameStats

from elfantasy.config import DATA_DIR
from elfantasy.functions import get_euroleague_data, standings_from_games, tidy_euroleague_data, tidy_games_data

competition_code = "E"
season = 2024

gamestats = GameStats(competition_code)
games_raw = gamestats.get_game_reports_single_season(season)
games = tidy_games_data(games_raw)
standings = standings_from_games(games)

df_raw = get_euroleague_data()
df = tidy_euroleague_data(df_raw)

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

player_averages.to_clipboard(index=False)

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
