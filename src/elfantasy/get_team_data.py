# https://github.com/giasemidis/euroleague_api/blob/main/notebooks/get-season-stats.ipynb
import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData

from elfantasy.config import DATA_DIR

competition_code = "E"
season = 2024
# season_start = 2023
# season_end = 2024

boxscore = BoxScoreData(competition_code)

tdf = boxscore.get_game_boxscore_quarter_data_single_season(season)
# tdf = boxscore.get_game_boxscore_quarter_data_multiple_seasons(season)

tdf.to_csv(DATA_DIR / f"team_data_{season}.csv", index=False)
