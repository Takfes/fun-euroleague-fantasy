import pandas as pd

from elfantasy.config import Configuration
from elfantasy.features import (
    calculate_game_codes,
    calculate_running_standings,
    calculate_standings,
    make_lineup_static_feats,
    make_player_static_feats,
    make_player_tempor_feats,
    tidy_euroleague_data,
    tidy_games_data,
)
from elfantasy.utils import get_timetag, read_datalog, update_datalog

# pandas settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

# timetag
timetag = get_timetag()

# configuration
config = Configuration()
data_dir_euroleague_games = config.data_dir_euroleague_games
data_dir_euroleague_data = config.data_dir_euroleague_data
data_dir_features_data = config.data_dir_features_data
limit_up_to_week = config.limit_up_to_week
datalog = read_datalog()

# read games data
euroleague_games_path = data_dir_euroleague_games / datalog["games"]
games_raw = pd.read_csv(euroleague_games_path)
# filter games data based on limit_up_to_week
games_raw_static = games_raw.loc[games_raw["Round"] <= limit_up_to_week]
# prepare games data
games = tidy_games_data(games_raw_static)
# calculate game codes
game_codes = calculate_game_codes(games)
# calculate standings
standings = calculate_standings(games)
# caluculate running standings
standings_running = calculate_running_standings(games, game_codes)

# read euroleague data
euroleague_data_path = data_dir_euroleague_data / datalog["data"]
data_raw = pd.read_csv(euroleague_data_path)
# filter euroleague data based on limit_up_to_week
data_raw_static = data_raw.loc[data_raw["week"] <= limit_up_to_week]
# prepare euroleague data
data = tidy_euroleague_data(data_raw_static, games, game_codes)

# feature engineering
df1 = make_lineup_static_feats(data, standings_running)
df2 = make_player_static_feats(df1)
df3 = make_player_tempor_feats(df2)

df_features = (
    df3.sort_values(by=["week", "team_code", "position", "slug"], ascending=True).reset_index(drop=True).copy()
)

# save features
features_file_name = f"features_{timetag}.csv"
df_features.to_csv(data_dir_features_data / features_file_name, index=False)

# update datalog
datalog_update = {"features": features_file_name}
update_datalog(datalog_update)
