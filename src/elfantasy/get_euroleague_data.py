from datetime import datetime

import pandas as pd
import requests

from elfantasy.config import DATA_DIR


def euroleague_data(season_id=17, stats_type="avg"):
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


if __name__ == "__main__":
    timetag = datetime.now().strftime("%Y%m%d%H%M%S")
    edf = euroleague_data()
    edf.to_csv(DATA_DIR / f"euroleague_data_{timetag}.csv", index=False)
