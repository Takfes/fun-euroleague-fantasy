from pathlib import Path

import pandas as pd

from elfantasy.acquisition import get_euroleague_data, get_euroleague_games
from elfantasy.config import Configuration
from elfantasy.utils import get_timetag, update_datalog

# configuration
config = Configuration(use_dotenv_config_yaml=True)
competition_code = config.competition_code
season = config.season
game_stats_dir = config.data_dir_euroleague_games
euroleague_data_dir = config.data_dir_euroleague_data

# timetag
timetag = get_timetag()

# gamestats
games_file_name = f"games_{timetag}.csv"
games = get_euroleague_games(competition_code, season)
games.to_csv(Path(game_stats_dir, games_file_name), index=False)

# euroleague data
data_file_name = f"data_{timetag}.csv"
data = get_euroleague_data()
data.to_csv(Path(euroleague_data_dir, data_file_name), index=False)

# update datalog
datalog_update = {"games": games_file_name, "data": data_file_name}
update_datalog(datalog_update)
