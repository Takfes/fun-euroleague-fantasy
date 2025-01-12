import pandas as pd
import requests
from euroleague_api.game_stats import GameStats

from elfantasy.utils import timeit


@timeit
def get_euroleague_games(competition_code, season):
    gamestats = GameStats(competition_code)
    games = gamestats.get_game_reports_single_season(season)
    return games


@timeit
def get_euroleague_data(season_id=17, stats_type="avg"):
    datasets = []
    week = 1
    while True:
        url = f"https://www.dunkest.com/api/stats/table?season_id={season_id}&mode=dunkest&stats_type={stats_type}&weeks%5B%5D={week}&rounds%5B%5D=1&rounds%5B%5D=2"
        response = requests.get(url, timeout=10)
        if response.status_code != 200 or not response.json():
            break
        datasets.append(pd.DataFrame(response.json()).assign(week=week))
        print(f"downloaded data for {season_id=} {week=}")
        week += 1
    return pd.concat(datasets, ignore_index=True)
