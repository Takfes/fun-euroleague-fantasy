# https://github.com/giasemidis/euroleague_api/blob/main/notebooks/get-season-stats.ipynb
import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData

from elfantasy.config import DATA_DIR

competition_code = "E"
season = 2024
# season_start = 2023
# season_end = 2024

boxscore = BoxScoreData(competition_code)

pdf = boxscore.get_player_boxscore_stats_single_season(season)
# pdf = boxscore.get_player_boxscore_stats_multiple_seasons(season_start, season_end)

pdf.to_csv(DATA_DIR / f"player_data_{season}.csv", index=False)
