import os
from pathlib import Path

from pyprojroot.here import here

DATA_DIR = here("data")

EUROLEAGUE_URL_TEMP = "https://www.dunkest.com/api/stats/table?season_id=17&mode=dunkest&stats_type=avg&weeks%5B%5D=19&rounds%5B%5D=1&rounds%5B%5D=2"

EUROLEAGUE_URL_2425 = "https://www.dunkest.com/api/stats/table?season_id=17&mode=nba&stats_type=tot&weeks%5B%5D=19&rounds%5B%5D=1&rounds%5B%5D=2&teams%5B%5D=31&teams%5B%5D=32&teams%5B%5D=33&teams%5B%5D=34&teams%5B%5D=35&teams%5B%5D=36&teams%5B%5D=37&teams%5B%5D=38&teams%5B%5D=39&teams%5B%5D=40&teams%5B%5D=41&teams%5B%5D=42&teams%5B%5D=43&teams%5B%5D=44&teams%5B%5D=45&teams%5B%5D=47&teams%5B%5D=48&teams%5B%5D=60&positions%5B%5D=1&positions%5B%5D=2&positions%5B%5D=3&player_search=&min_cr=4&max_cr=35&sort_by=pdk&sort_order=desc&iframe=yes&date_from=2024-10-03&date_to=2025-05-31"

EUROLEAGUE_URL_2324 = "https://www.dunkest.com/api/stats/table?season_id=15&mode=nba&stats_type=tot&weeks%5B%5D=43&rounds%5B%5D=1&teams%5B%5D=31&teams%5B%5D=32&teams%5B%5D=33&teams%5B%5D=34&teams%5B%5D=35&teams%5B%5D=36&teams%5B%5D=37&teams%5B%5D=38&teams%5B%5D=39&teams%5B%5D=40&teams%5B%5D=41&teams%5B%5D=42&teams%5B%5D=43&teams%5B%5D=44&teams%5B%5D=45&teams%5B%5D=46&teams%5B%5D=47&teams%5B%5D=48&positions%5B%5D=1&positions%5B%5D=2&positions%5B%5D=3&player_search=&min_cr=4&max_cr=35&sort_by=pdk&sort_order=desc&iframe=yes&date_from=2023-10-05&date_to=2024-05-31"

EUROLEAGUE_URL_2223 = "https://www.dunkest.com/api/stats/table?season_id=11&mode=nba&stats_type=tot&weeks%5B%5D=41&rounds%5B%5D=1&teams%5B%5D=31&teams%5B%5D=32&teams%5B%5D=33&teams%5B%5D=34&teams%5B%5D=35&teams%5B%5D=36&teams%5B%5D=37&teams%5B%5D=38&teams%5B%5D=39&teams%5B%5D=40&teams%5B%5D=41&teams%5B%5D=42&teams%5B%5D=43&teams%5B%5D=44&teams%5B%5D=45&teams%5B%5D=46&teams%5B%5D=47&teams%5B%5D=48&positions%5B%5D=1&positions%5B%5D=2&positions%5B%5D=3&player_search=&min_cr=4&max_cr=35&sort_by=pdk&sort_order=desc&iframe=yes&date_from=2022-10-06&date_to=2023-05-21"

EUROLEAGUE_URL = EUROLEAGUE_URL_2425
