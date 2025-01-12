import numpy as np
import pandas as pd

from elfantasy.utils import softmax


def tidy_euroleague_data(df, games, game_codes):
    # dfc = df_raw.copy()
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
        "slug",
        "first_name",
        "last_name",
        "id",
        "position",
        "position_id",
        "team_code",
        "team_name",
        "team_id",
        "week",
        "game_code",
        "home_away",
        "hometeamcode",
        "awayteamcode",
        "pdk",
        "cr",
        "plus",
        "min",
        "starter",
        "offensive_stats",
        "defensive_stats",
        "plus_minus",
        "valuation_plus",
        "valuation_minus",
        "valuation",
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
    ]

    # https://euroleaguefantasy.euroleaguebasketball.net/10/rules
    valuation_plus = ["pts", "reb", "ast", "stl", "blk", "fouls_received"]
    valuation_minus = ["tov", "blka", "pf", "fgl", "tpl", "ftl"]

    defensive_stats = ["reb", "stl", "blk", "blka", "dreb", "fouls_received"]
    offensive_stats = ["pts", "ast", "oreb"]

    # apply dtypes on the dataframe
    dfc = dfc.astype(dtypes)
    dfc = dfc.drop(columns=drop_columns)

    dfc["fgl"] = dfc["fga"] - dfc["fgm"]
    dfc["tpl"] = dfc["tpa"] - dfc["tpm"]
    dfc["ftl"] = dfc["fta"] - dfc["ftm"]

    dfc["offensive_stats"] = dfc[offensive_stats].sum(axis=1)
    dfc["defensive_stats"] = dfc[defensive_stats].sum(axis=1)
    dfc["valuation_plus"] = dfc[valuation_plus].sum(axis=1)
    dfc["valuation_minus"] = dfc[valuation_minus].sum(axis=1)
    dfc["valuation"] = dfc["valuation_plus"] - dfc["valuation_minus"]

    # bring game_code info in df
    dfc = (
        dfc.merge(game_codes, left_on=["week", "team_code"], right_on=["Round", "team"])
        .drop(columns=["Round", "team"])
        .rename(columns={"Gamecode": "game_code"})
        .assign(home_away=lambda x: x["home_away"].replace({"HomeTeamCode": "H", "AwayTeamCode": "A"}))
    )

    # bringn in home and away team names
    dfc = dfc.merge(
        games[["Gamecode", "HomeTeamCode", "AwayTeamCode"]].rename(columns=str.lower),
        left_on="game_code",
        right_on="gamecode",
    )

    return dfc[column_order]


def tidy_games_data(df):
    dfc = df.copy()

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

    columns_selection = [
        "Season",
        "Gamecode",
        "Round",
        "Turn",
        "Date",
        "DateTime",
        "PhaseDesc",
        "HomeTeamName",
        "HomeTeamCode",
        "HomeTeamScore",
        "AwayTeamName",
        "AwayTeamCode",
        "AwayTeamScore",
    ]

    # dfc = games_raw.copy()
    dfc = dfc.rename(columns=columns_mapping)
    # add date column
    dfc["Date"] = pd.to_datetime(dfc["DateTime"]).dt.date
    # add turn column
    dfc["Turn"] = dfc.groupby(["Season", "Round"])["Date"].transform("rank", method="dense").astype(int)

    return dfc[columns_selection]


def calculate_standings(df, streak=(1, 3, 5)):
    dfc = df.copy()
    # dfc = temp.copy() # from the calculate_running_standings

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


def calculate_running_standings(games, game_codes):
    # iterate to create running standings list of dataframe
    _rounds = sorted(games.Round.unique().tolist())
    standings_data = []
    for r in _rounds:
        temp = games[games.Round <= r]
        standings_data.append(calculate_standings(temp).assign(Round=r))

    # combine all dataframes into one
    standings_running = (
        pd.concat(standings_data, ignore_index=True)
        .sort_values(["Round", "Won", "PointsDiff"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    # bring game_code info in standings
    standings_running_gc = (
        standings_running.merge(game_codes, left_on=["Round", "TeamCode"], right_on=["Round", "team"])
        .assign(HomeAway=lambda x: x["home_away"].replace({"HomeTeamCode": "H", "AwayTeamCode": "A"}))
        .drop(columns=["team", "home_away"])
    ).fillna(0)

    # there are some missing values in the standings due to cancelled(?) games
    teams = standings_running_gc.TeamCode.unique()
    rounds = standings_running_gc.Round.unique()
    # create dataframe with combination of all teams and all rounds
    standings_running_full = pd.DataFrame(
        [(r, t) for r in rounds for t in teams],
        columns=["Round", "TeamCode"],
    ).merge(standings_running_gc, how="left", on=["Round", "TeamCode"])
    # Forward fill null values per group of TeamCodes
    standings_running_full = (
        standings_running_full.groupby("TeamCode", group_keys=False)
        .apply(lambda group: group.ffill())
        .reset_index(drop=True)
    )

    # reorder columns to have 'Round', 'Gamecode', 'HomeAway' at the front
    cols = ["Round", "Gamecode", "HomeAway"] + [
        col for col in standings_running_full.columns if col not in ["Round", "Gamecode", "HomeAway"]
    ]
    return standings_running_full[cols]


def calculate_game_codes(games):
    return games.melt(
        id_vars=["Round", "Gamecode"],
        value_vars=["HomeTeamCode", "AwayTeamCode"],
        var_name="home_away",
        value_name="team",
    )


def make_lineup_static_feats(df, standings_running):
    """
    This function takes in a DataFrame of game data and a DataFrame of running standings, and returns a new DataFrame with additional features related to team form. It merges the home and away team standings with the game data, calculates various differences in performance metrics between the home and away teams, and renames the new columns with a suffix to indicate they were created by this function.
    """

    dfc = df.copy()
    function_suffix = "lnp_sttc"
    # select features to maintain out of running standings dataset
    standings_running_features = [
        "Round",
        "TeamCode",
        "Won",
        "HomeWinRate",
        "AwayWinRate",
        "WinRate",
        "WinsLast1Games",
        "WinsLast3Games",
        "WinsLast5Games",
    ]
    # create home and away standings
    # ! x.Round + 1 --> is to bind with the next week's game and prevent data leakage
    # home standings
    home_standings = (
        standings_running[standings_running_features]
        .assign(Round=lambda x: x.Round + 1)
        .rename(columns={c: f"HomeTeam_{c}" for c in standings_running_features if c not in ["Round", "TeamCode"]})
    )
    # away standings
    away_standings = (
        standings_running[standings_running_features]
        .assign(Round=lambda x: x.Round + 1)
        .rename(columns={c: f"AwayTeam_{c}" for c in standings_running_features if c not in ["Round", "TeamCode"]})
    )
    # merge home and away standings
    team_form = (
        dfc.merge(home_standings, left_on=["week", "hometeamcode"], right_on=["Round", "TeamCode"], how="left")
        .drop(columns=["Round", "TeamCode"])
        .merge(away_standings, left_on=["week", "awayteamcode"], right_on=["Round", "TeamCode"], how="left")
        .fillna(0)
        .assign(home_away_factor=lambda x: np.where(x["home_away"] == "H", 1, -1))
        .assign(
            win_diff=lambda x: x["home_away_factor"] * (x["HomeTeam_Won"] - x["AwayTeam_Won"]),
            win_rate_diff=lambda x: x["home_away_factor"] * (x["HomeTeam_WinRate"] - x["AwayTeam_WinRate"]),
            win_rate_ha_diff=lambda x: x["home_away_factor"] * (x["HomeTeam_HomeWinRate"] - x["AwayTeam_AwayWinRate"]),
            win_last1games=lambda x: x["home_away_factor"]
            * (x["HomeTeam_WinsLast1Games"] - x["AwayTeam_WinsLast1Games"]),
            win_last3games=lambda x: x["home_away_factor"]
            * (x["HomeTeam_WinsLast3Games"] - x["AwayTeam_WinsLast3Games"]),
            win_last5games=lambda x: x["home_away_factor"]
            * (x["HomeTeam_WinsLast5Games"] - x["AwayTeam_WinsLast5Games"]),
        )
        .drop(
            columns=[
                "Round",
                "TeamCode",
                "HomeTeam_AwayWinRate",
                "AwayTeam_HomeWinRate",
                "HomeTeam_Won",
                "AwayTeam_Won",
                "HomeTeam_WinRate",
                "AwayTeam_WinRate",
                "HomeTeam_HomeWinRate",
                "AwayTeam_AwayWinRate",
                "HomeTeam_WinsLast1Games",
                "HomeTeam_WinsLast3Games",
                "HomeTeam_WinsLast5Games",
                "AwayTeam_WinsLast3Games",
                "AwayTeam_WinsLast1Games",
                "AwayTeam_WinsLast5Games",
            ]
        )
        .rename(columns=lambda x: x.lower())
    )

    # bring in team's total valuation and is_winner
    team_stats = (
        dfc.groupby(["game_code", "team_code"], as_index=False)
        .agg(team_pts=("pts", "sum"), team_valuation=("valuation", "sum"))
        .assign(max_pts_per_gc=lambda x: x.groupby("game_code")["team_pts"].transform("max"))
        .assign(is_winner=lambda x: (x["team_pts"] == x["max_pts_per_gc"]).astype(int))
        .assign(team_valuation_lag_1=lambda x: x.groupby("team_code")["team_valuation"].shift())
        .assign(
            team_valuation_expanding=lambda x: x.sort_values(by="game_code", ascending=True)
            .groupby("team_code")["team_valuation"]
            .apply(lambda x: x.shift().expanding().mean())
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .drop(columns=["max_pts_per_gc"])
    )

    team_form_stats = team_form.merge(team_stats, on=["game_code", "team_code"], how="left")

    # identify newly created columns
    new_columns = [x for x in team_form_stats if x not in df.columns]
    new_columns_names = {x: f"{x}_{function_suffix}" for x in new_columns}

    return team_form_stats.rename(columns=new_columns_names)


def make_player_static_feats(df, features=None):
    """
    Calculate player contributions against their team, per week (isolated calculation - figures per week concern that week only) and for specified features in a DataFrame. This function takes a DataFrame containing player statistics and calculates player contributions for specified features. The contributions are calculated in two ways:
    1. As a softmax of the feature values within each position, team, and week.
    2. As a percentage of the total feature values within each team and week. The function also shifts the calculated contributions to prevent data leakage.
    """

    # preparatory steps
    dfc = df.copy()
    function_suffix = "plr_sttc"

    # features for which to calculate contribution
    if features is None:
        features = [
            "plus_minus",
            "valuation",
            "min",
        ]

    # apply contribution features
    for feat in features:
        # calculate rankings per week, position, team
        dfc[f"{feat}_rank_week_pos_team"] = dfc.groupby(["week", "position", "team_name"])[feat].rank(
            ascending=False, method="min"
        )

        dfc[f"{feat}_rank_week_team"] = dfc.groupby(["week", "team_name"])[feat].rank(ascending=False, method="min")

        dfc[f"{feat}_rank_week_pos"] = dfc.groupby(["week", "position"])[feat].rank(ascending=False, method="min")

        dfc[f"{feat}_rank_week"] = dfc.groupby(["week"])[feat].rank(ascending=False, method="min")

        # player contribution per position - as softmax
        dfc[f"{feat}_contrib_pos_sft"] = dfc.groupby(["week", "team_name", "team_code", "position"], as_index=False)[
            feat
        ].transform(softmax)
        # ! shift to prevent data leakage
        dfc[f"{feat}_contrib_pos_sft"] = (
            dfc.sort_values(by=["week"], ascending=True)
            .groupby(["slug"], as_index=False)[f"{feat}_contrib_pos_sft"]
            .shift()
        )

        # player contribution per team - as pct over total
        dfc[f"{feat}_contrib_ttl_agg"] = dfc[feat] / dfc.groupby(["week", "team_name", "team_code"], as_index=False)[
            feat
        ].transform("sum")
        # ! shift to prevent data leakage
        dfc[f"{feat}_contrib_ttl_agg"] = (
            dfc.sort_values(by=["week"], ascending=True)
            .groupby(["slug"], as_index=False)[f"{feat}_contrib_ttl_agg"]
            .shift()
        )

    # identify newly created columns
    new_columns = [x for x in dfc if x not in df.columns]
    new_columns_names = {x: f"{x}_{function_suffix}" for x in new_columns}

    return dfc.fillna(0).rename(columns=new_columns_names)


def make_player_tempor_feats(df, features=None, lags=(1, 3), rolls=(3,)):
    """
    This function calculates rolling statistics for player performance data in a DataFrame. It creates new features based on lagged values and rolling averages for specified columns.
    """

    # preparatory steps
    dfc = df.copy()
    function_suffix = "plr_tmpr"

    # infer features to apply rolling stats
    if features is None:
        features = [x for x in dfc.columns if x == "valuation" or "contrib" in x]

    # apply rolling stats
    for feature in features:
        # expanding (ytd) average
        dfc[f"{feature}_expanding_mean"] = (
            dfc.sort_values(by=["week"], ascending=True)
            .groupby(["slug"])[feature]
            .apply(lambda x: x.shift().expanding().mean())
            .sort_index(level=1)
            .reset_index(drop=True)
        )

        # expanding (ytd) std
        dfc[f"{feature}_expanding_std"] = (
            dfc.sort_values(by=["week"], ascending=True)
            .groupby(["slug"])[feature]
            .apply(lambda x: x.shift().expanding().std())
            .sort_index(level=1)
            .reset_index(drop=True)
        )

        # lag features
        for lag in lags:
            # create lag feature
            dfc[f"{feature}_lag_{lag}"] = (
                dfc.sort_values(by=["week"], ascending=True).groupby(["slug"])[f"{feature}"].shift(lag)
            )

        # rolling features
        for roll in rolls:
            # create rolling average feature
            dfc[f"{feature}_roll_{roll}"] = (
                dfc.sort_values(by=["week"], ascending=True)
                .groupby(["slug"])[f"{feature}"]
                .rolling(roll, closed="left")
                .mean()
                .reset_index(0, drop=True)
            )

    # identify newly created columns
    new_columns = [x for x in dfc if x not in df.columns]
    new_columns_names = {x: f"{x}_{function_suffix}" for x in new_columns}

    return dfc.fillna(0).rename(columns=new_columns_names)
