import os
from pathlib import Path

import pandas as pd

from elfantasy.config import DATA_DIR
from elfantasy.get_euroleague_data import euroleague_data

"""
# ==============================================================
# Load Data
# ==============================================================
"""

# list all files in the data directory
datafiles = os.listdir(DATA_DIR)

# filter files by type
playerfiles = [f for f in datafiles if f.startswith("player_data_")]
teamfiles = [f for f in datafiles if f.startswith("team_data_")]

# load players data
players = (
    pd.concat([pd.read_csv(Path(DATA_DIR) / f) for f in playerfiles], ignore_index=True)
    .sort_values(by=["Season", "Gamecode", "Player_ID"], ascending=[True, True, True])
    .reset_index(drop=True)
)

# load teams data
teams = (
    pd.concat([pd.read_csv(Path(DATA_DIR) / f) for f in teamfiles], ignore_index=True)
    .sort_values(by=["Season", "Gamecode", "Team"], ascending=[True, True, True])
    .reset_index(drop=True)
)

# load euroleague data
edf = euroleague_data()

"""
# ==============================================================
# Data Processing Player Euroleague Files Data
# ==============================================================
"""

edf["Player"] = edf["last_name"].str.upper() + ", " + edf["first_name"].str.upper()

player_position_mapping = edf[["Player", "position"]].drop_duplicates().set_index("Player").to_dict().get("position")

player_credits_mapping = (
    edf[["Player", "cr", "week"]]
    .sort_values(by=["week"], ascending=False)
    .drop_duplicates(keep="first")
    .drop(columns=["week"])
    .set_index("Player")
    .to_dict()
    .get("cr")
)

edf.to_clipboard(index=False)

"""
# ==============================================================
# Data Processing Player Data
# ==============================================================
"""


# Define Functions
def minutes_to_float(minutes):
    try:
        return float(minutes.split(":")[0]) + float(minutes.split(":")[1]) / 60
    except Exception:
        return 0


# Define Constants and Parameters
phase_mapping = {"RS": "Regular Season", "PI": "Play-In", "PO": "Play-Offs", "FF": "Final Four"}

valuation_plus = ["Points", "TotalRebounds", "Steals", "Assistances", "BlocksFavour", "FoulsReceived"]

valuation_minus = [
    "Turnovers",
    "BlocksAgainst",
    "FoulsCommited",
    "FieldGoalsMissed2",
    "FieldGoalsMissed3",
    "FreeThrowsMissed",
]

basic_columns = [
    "Season",
    "Phase",
    "PhaseDescription",
    "Round",
    "Gamecode",
    "Home",
    "Team",
    "Win",
    "Points",
    "FieldGoalsMade2",
    "FieldGoalsAttempted2",
    "FieldGoalsMissed2",
    "FieldGoalsMade3",
    "FieldGoalsAttempted3",
    "FieldGoalsMissed3",
    "FreeThrowsMade",
    "FreeThrowsAttempted",
    "FreeThrowsMissed",
    "OffensiveRebounds",
    "DefensiveRebounds",
    "TotalRebounds",
    "Assistances",
    "Steals",
    "Turnovers",
    "BlocksFavour",
    "BlocksAgainst",
    "FoulsCommited",
    "FoulsReceived",
    "ValuationPlus",
    "ValuationMinus",
    "Valuation",
    "ValuationCalculated",
]

# Adjust existing columns
players["Player"] = players["Player"].str.upper()
players["Player_ID"] = players["Player_ID"].str.strip()

# New Columns
players["Week"] = players.groupby(["Season", "Team"])["Gamecode"].rank(method="dense").astype(int)
players["Position"] = players["Player"].map(player_position_mapping)
players["MinutesCalculated"] = players["Minutes"].apply(minutes_to_float)
players["FieldGoalsMissed2"] = players["FieldGoalsAttempted2"] - players["FieldGoalsMade2"]
players["FieldGoalsMissed3"] = players["FieldGoalsAttempted3"] - players["FieldGoalsMade3"]
players["FreeThrowsMissed"] = players["FreeThrowsAttempted"] - players["FreeThrowsMade"]
players["ValuationPlus"] = players[valuation_plus].sum(axis=1)
players["ValuationMinus"] = players[valuation_minus].sum(axis=1)
players["ValuationCalculated"] = players["ValuationPlus"] - players["ValuationMinus"]
players["PhaseDescription"] = players["Phase"].map(phase_mapping)


# Splits dataframes into players and teams
# Players
pdf = players.loc[~players.Player_ID.isin(["Team", "Total"]), :].reset_index(drop=True)

# Teams
tdf = players.loc[players.Player_ID == "Total", :].reset_index(drop=True)

tdf["Win"] = (tdf.groupby(["Season", "Gamecode"])["Points"].transform("max") == tdf["Points"]).astype(int)


# Review
players.columns
pdf.to_clipboard(index=False)
tdf[team_columns].to_clipboard(index=False)
