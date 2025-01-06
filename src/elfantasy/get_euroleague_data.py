from datetime import datetime

import pandas as pd
import requests

from elfantasy.config import DATA_DIR, EUROLEAGUE_URL_2223, EUROLEAGUE_URL_2324, EUROLEAGUE_URL_2425

EUROLEAGUE_URL = EUROLEAGUE_URL_2425
timetag = datetime.now().strftime("%Y%m%d%H%M%S")

response = requests.get(EUROLEAGUE_URL, timeout=10)
data = response.json()
edf = pd.DataFrame(data)

edf.to_csv(DATA_DIR / f"euroleague_data_{timetag}.csv", index=False)
