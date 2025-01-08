import numpy as np
import pandas as pd
import requests


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


def tidy_euroleague_data(df):
    dfc = df.copy()

    dtypes = {
        "cr": "float64",
        "pdk": "float64",
        "plus": "float64",
        "min": "float64",
        "fgp": "float64",
        "tpp": "float64",
        "ftp": "float64",
        "starter": "float64",
        "pts": "float64",
        "ast": "float64",
        "reb": "float64",
        "stl": "float64",
        "blk": "float64",
        "blka": "float64",
        "fgm": "float64",
        "fgm_tot": "float64",
        "fga": "float64",
        "fga_tot": "float64",
        "tpm": "float64",
        "tpm_tot": "float64",
        "tpa": "float64",
        "tpa_tot": "float64",
        "ftm": "float64",
        "ftm_tot": "float64",
        "fta": "float64",
        "fta_tot": "float64",
        "oreb": "float64",
        "dreb": "float64",
        "tov": "float64",
        "pf": "float64",
        "fouls_received": "float64",
        "plus_minus": "float64",
    }

    drop_columns = [
        "gp",
        "fgm_tot",
        "fga_tot",
        "tpm_tot",
        "tpa_tot",
        "ftm_tot",
        "fta_tot",
    ]

    column_order = [
        "id",
        "week",
        "first_name",
        "last_name",
        "slug",
        "team_id",
        "team_code",
        "team_name",
        "position_id",
        "position",
        "cr",
        "pdk",
        "plus",
        "min",
        "starter",
        "pts",
        "fgm",
        "fga",
        "fgl",
        "fgp",
        "tpm",
        "tpa",
        "tpl",
        "tpp",
        "ftm",
        "fta",
        "ftl",
        "ftp",
        "ast",
        "reb",
        "stl",
        "blk",
        "blka",
        "oreb",
        "dreb",
        "tov",
        "pf",
        "fouls_received",
        "plus_minus",
        "valuation_plus",
        "valuation_minus",
        "valuation",
    ]

    # https://euroleaguefantasy.euroleaguebasketball.net/10/rules
    valuation_plus = ["pts", "reb", "ast", "stl", "blk", "fouls_received"]
    valuation_minus = ["tov", "blka", "pf", "fgl", "tpl", "ftl"]

    # apply dtypes on the dataframe
    dfc = dfc.astype(dtypes)
    dfc = dfc.drop(columns=drop_columns)

    dfc["fgl"] = dfc["fga"] - dfc["fgm"]
    dfc["tpl"] = dfc["tpa"] - dfc["tpm"]
    dfc["ftl"] = dfc["fta"] - dfc["ftm"]

    dfc["valuation_plus"] = dfc[valuation_plus].sum(axis=1)
    dfc["valuation_minus"] = dfc[valuation_minus].sum(axis=1)
    dfc["valuation"] = dfc["valuation_plus"] - dfc["valuation_minus"]

    return dfc[column_order]


def tidy_games_data(df):
    dfc = df.copy()

    columns_selection = [
        "Season",
        "Gamecode",
        "Round",
        "Turn",
        "Date",
        "utcDate",
        "group.rawName",
        "local.club.name",
        "local.club.tvCode",
        "local.score",
        "road.club.name",
        "road.club.tvCode",
        "road.score",
    ]

    columns_mapping = {
        "Season": "Season",
        "Gamecode": "Gamecode",
        "Round": "Round",
        "utcDate": "DateTime",
        "group.rawName": "PhaseDesc",
        "local.club.name": "HomeTeamName",
        "local.club.tvCode": "HomeTeamCode",
        "local.score": "HomeTeamScore",
        "road.club.name": "AwayTeamName",
        "road.club.tvCode": "AwayTeamCode",
        "road.score": "AwayTeamScore",
    }

    # dfc = games_raw.copy()
    dfc["Date"] = pd.to_datetime(dfc["utcDate"]).dt.date
    dfc["Turn"] = dfc.groupby(["Season", "Round"])["Date"].transform("rank", method="dense").astype(int)

    return dfc[columns_selection].rename(columns=columns_mapping)


def standings_from_games(df, streak=(1, 3, 5)):
    dfc = df.copy()

    def coach_scoring(points):
        if points > 0:
            if points <= 10:
                return 10
            elif points <= 20:
                return 20
            else:
                return 25
        else:
            if points >= -10:
                return -5
            elif points >= -20:
                return -10
            else:
                return -20

    # dfc = games.copy()
    dfc["HomeScoreDiff"] = dfc["HomeTeamScore"] - dfc["AwayTeamScore"]
    dfc["AwayScoreDiff"] = dfc["AwayTeamScore"] - dfc["HomeTeamScore"]
    dfc["HomeTeamWin"] = np.where(dfc["HomeScoreDiff"] > 0, 1, 0)
    dfc["AwayTeamWin"] = np.where(dfc["AwayScoreDiff"] > 0, 1, 0)
    dfc["HomeCoachScore"] = dfc["HomeScoreDiff"].apply(coach_scoring)
    dfc["AwayCoachScore"] = dfc["AwayScoreDiff"].apply(coach_scoring)

    homegames = (
        dfc.filter(regex="^(?!.*Away).*$", axis=1)
        .rename(columns=lambda x: x.replace("Home", ""))
        .assign(Venue_is_home=1)
    )
    awaygames = (
        dfc.filter(regex="^(?!.*Home).*$", axis=1)
        .rename(columns=lambda x: x.replace("Away", ""))
        .assign(Venue_is_home=-1)
    )
    games_long = pd.concat([homegames, awaygames], ignore_index=True).assign(
        VenueWL=lambda x: x.Venue_is_home * x.TeamWin
    )

    standings = (
        games_long.sort_values(by=["Season", "Gamecode"])
        .groupby(["TeamName", "TeamCode"], as_index=False)
        .agg(
            CoachScore=("CoachScore", "sum"),
            PointsDiff=("ScoreDiff", "sum"),
            PointsPlus=("ScoreDiff", lambda x: x[x > 0].sum()),
            PointsMinus=("ScoreDiff", lambda x: x[x < 0].sum()),
            GamesPlayed=("TeamCode", "count"),
            Won=("TeamWin", lambda x: (x == 1).sum()),
            Lost=("TeamWin", lambda x: (x != 1).sum()),
            HomeWins=("VenueWL", lambda x: x[x > 0].sum()),
            AwayWins=("VenueWL", lambda x: x[x < 0].count()),
            HomeGames=("Venue_is_home", lambda x: x[x > 0].sum()),
            AwayGames=("Venue_is_home", lambda x: x[x < 0].count()),
            LastGames=("TeamWin", list),
        )
        .assign(
            HomeLosses=lambda x: x.HomeGames - x.HomeWins,
            AwayLosses=lambda x: x.AwayGames - x.AwayWins,
            HomeWinRate=lambda x: round((x.HomeWins / x.HomeGames) * 100, 2),
            AwayWinRate=lambda x: round((x.AwayWins / x.AwayGames) * 100, 2),
            WinRate=lambda x: round((x.Won / x.GamesPlayed) * 100, 2),
        )
        .sort_values(by="Won", ascending=False)
        .reset_index(drop=True)
    )

    column_order = [
        "TeamName",
        "TeamCode",
        "CoachScore",
        "GamesPlayed",
        "Won",
        "Lost",
        "PointsDiff",
        "PointsPlus",
        "PointsMinus",
        "HomeGames",
        "AwayGames",
        "HomeWins",
        "AwayWins",
        "HomeLosses",
        "AwayLosses",
        "HomeWinRate",
        "AwayWinRate",
        "WinRate",
        "LastGames",
    ]

    for s in streak:
        colname = f"WinsLast{s}Games"
        standings[colname] = standings["LastGames"].apply(lambda x: sum(x[-s:]) if len(x) >= s else 0)
        column_order.append(colname)

    return standings[column_order].drop(columns=["LastGames"])
