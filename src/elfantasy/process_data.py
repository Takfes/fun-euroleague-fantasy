import os
from pathlib import Path

import pandas as pd
from regex import E

from elfantasy.config import DATA_DIR

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

# get the latest euroleague file
eulfiles = [f for f in datafiles if f.startswith("euroleague_data_")]
eulfile_ids = [f.split("_")[-1].split(".")[0] for f in eulfiles]
eulfile_latest_id = str(max(map(int, eulfile_ids)))
eulfile_latest = [x for x in eulfiles if eulfile_latest_id in x][0]

# load players data
pdf = pd.concat([pd.read_csv(Path(DATA_DIR) / f) for f in playerfiles], ignore_index=True).sort_values(
    by=["Season", "Gamecode", "Player_ID"], ascending=[True, True, True]
)

# load teams data
tdf = pd.concat([pd.read_csv(Path(DATA_DIR) / f) for f in teamfiles], ignore_index=True).sort_values(
    by=["Season", "Gamecode", "Team"], ascending=[True, True, True]
)

# load euroleague data
edf = pd.read_csv(Path(DATA_DIR) / eulfile_latest)

# load euroleagues data
ldf = pd.concat([pd.read_csv(Path(DATA_DIR) / f) for f in eulfiles], ignore_index=True)


"""
# ==============================================================
# Data Processing Player Euroleague Files Data
# ==============================================================
"""

ldf["Player"] = ldf["last_name"].str.upper() + ", " + ldf["first_name"].str.upper()

player_position_mapping = ldf[["Player", "position"]].drop_duplicates().set_index("Player").to_dict().get("position")

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

team_columns = [
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
pdf["Player"] = pdf["Player"].str.upper()
pdf["Player_ID"] = pdf["Player_ID"].str.strip()

# New Columns
pdf["Position"] = pdf["Player"].map(player_position_mapping)
pdf["MinutesCalculated"] = pdf["Minutes"].apply(minutes_to_float)
pdf["FieldGoalsMissed2"] = pdf["FieldGoalsAttempted2"] - pdf["FieldGoalsMade2"]
pdf["FieldGoalsMissed3"] = pdf["FieldGoalsAttempted3"] - pdf["FieldGoalsMade3"]
pdf["FreeThrowsMissed"] = pdf["FreeThrowsAttempted"] - pdf["FreeThrowsMade"]
pdf["ValuationPlus"] = pdf[valuation_plus].sum(axis=1)
pdf["ValuationMinus"] = pdf[valuation_minus].sum(axis=1)
pdf["ValuationCalculated"] = pdf["ValuationPlus"] - pdf["ValuationMinus"]
pdf["PhaseDescription"] = pdf["Phase"].map(phase_mapping)


team_pdf = (
    pdf.loc[pdf.Player == "TOTAL", :]
    .reset_index(drop=True)
    .sort_values(by=["Season", "Gamecode", "Team", "Home"], ascending=[True, True, True, True])
    .copy()
)

team_pdf["Win"] = (team_pdf.groupby(["Season", "Gamecode"])["Points"].transform("max") == team_pdf["Points"]).astype(
    int
)

team_pdf[team_columns].to_clipboard(index=False)

# pdf_grouped = (
#     pdf.loc[pdf.Player == "TOTAL", :]
#     .groupby(["Season", "Gamecode", "Team", "Home"])
#     .agg(
#         Points=("Points", "sum"),
#         TotalRebounds=("TotalRebounds", "sum"),
#         Steals=("Steals", "sum"),
#         Assistances=("Assistances", "sum"),
#         BlocksFavour=("BlocksFavour", "sum"),
#         FoulsReceived=("FoulsReceived", "sum"),
#         Turnovers=("Turnovers", "sum"),
#         BlocksAgainst=("BlocksAgainst", "sum"),
#         FoulsCommited=("FoulsCommited", "sum"),
#         FieldGoalsMissed2=("FieldGoalsMissed2", "sum"),
#         FieldGoalsMissed3=("FieldGoalsMissed3", "sum"),
#         FreeThrowsMissed=("FreeThrowsMissed", "sum"),
#         ValuationPlus=("ValuationPlus", "sum"),
#         ValuationMinus=("ValuationMinus", "sum"),
#         ValuationCalculated=("ValuationCalculated", "sum"),
#     )
#     .reset_index()
# )

# Review
pdf.columns
pdf.to_clipboard(index=False)
