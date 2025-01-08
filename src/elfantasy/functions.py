import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import requests
import seaborn as sns


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


@timeit
def optimize_team(data, value_col, budget=100, guards=4, forwards=4, centers=2):
    df = data.copy()
    max_players = guards + forwards + centers

    # prepare data
    slugs = df["slug"].unique()
    positions = df["position"].unique()
    posg = df[df.position == "G"]["slug"].unique()
    posf = df[df.position == "F"]["slug"].unique()
    posc = df[df.position == "C"]["slug"].unique()
    values = df[["slug", value_col]].fillna(0).set_index("slug").to_dict()[value_col]
    costs = df[["slug", "cr"]].fillna(0).set_index("slug").to_dict()["cr"]

    # Create a model
    model = pyo.ConcreteModel()

    # Define the sets
    model.P = pyo.Set(initialize=slugs)
    model.pos = pyo.Set(initialize=positions)
    model.posg = pyo.Set(initialize=posg)
    model.posf = pyo.Set(initialize=posf)
    model.posc = pyo.Set(initialize=posc)

    # Define parameters
    model.v = pyo.Param(model.P, initialize=values)
    model.c = pyo.Param(model.P, initialize=costs)
    model.max_players = pyo.Param(initialize=max_players)
    model.guards = pyo.Param(initialize=guards)
    model.forwards = pyo.Param(initialize=forwards)
    model.centers = pyo.Param(initialize=centers)
    model.budget = pyo.Param(initialize=budget)

    # Define decision variables
    model.x = pyo.Var(model.P, within=pyo.Binary)

    # Define constraints
    def cstr_max_players(model):
        return sum(model.x[p] for p in model.P) == model.max_players

    def cstr_guards(model):
        return sum(model.x[p] for p in model.posg) == model.guards

    def cstr_forwards(model):
        return sum(model.x[p] for p in model.posf) == model.forwards

    def cstr_centers(model):
        return sum(model.x[p] for p in model.posc) == model.centers

    def cstr_budget(model):
        return sum(model.x[p] * model.c[p] for p in model.P) <= model.budget

    model.cstr_max_players = pyo.Constraint(rule=cstr_max_players)
    model.cstr_guards = pyo.Constraint(rule=cstr_guards)
    model.cstr_forwards = pyo.Constraint(rule=cstr_forwards)
    model.cstr_centers = pyo.Constraint(rule=cstr_centers)
    model.cstr_budget = pyo.Constraint(rule=cstr_budget)

    # Define objective function
    def obj(model):
        return sum(model.x[p] * model.v[p] for p in model.P)

    model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    # solver options ------------------------------------------------
    solver = pyo.SolverFactory("appsi_highs")

    solver.options["tee"] = False
    solver.options["parallel"] = "on"
    solver.options["time_limit"] = 3600 / 2  # 30 minutes time limit
    solver.options["presolve"] = "on"
    solver.options["mip_rel_gap"] = 0.01  # 1% relative gap
    solver.options["simplex_strategy"] = 1  # Dual simplex
    solver.options["simplex_max_concurrency"] = 8  # Max concurrency
    solver.options["mip_min_logging_interval"] = 10  # Log every 10 seconds
    solver.options["mip_heuristic_effort"] = 0.2  # Increase heuristic effort
    solver.options["log_file"] = (
        "highs.log"  # Sometimes HiGHS doesn't update the console as it solves, so write log file too
    )
    # solver options ------------------------------------------------

    solver.solve(model, tee=False)

    solution = {k: int(v > 0.55) for k, v in model.x.extract_values().items()}
    df["solution"] = df["slug"].map(solution)
    df1 = df[df.solution == 1]

    # validation ------------------------------------------------
    assert df1.shape[0] == max_players, f"More than {max_players} players selected"

    assert df1[df1.position == "G"].slug.nunique() == guards, f"More than {guards} guards selected"

    assert df1[df1.position == "F"].slug.nunique() == forwards, f"More than {forwards} forwards selected"

    assert df1[df1.position == "C"].slug.nunique() == centers, f"More than {centers} centers selected"

    assert df1.cr.sum().item() <= budget + 0.001, f"Budget exceeded: {df1.cr.sum().item()} > {budget}"

    assert df1.groupby("team_code").size().max().item() <= 6, "More than 6 players from the same team selected"
    # end validation --------------------------------------------

    # objective_value = model.obj()
    objective_value = df[(df.solution == 1)].valuation.sum().item()
    return df1.slug.unique().tolist(), objective_value, df1.cr.sum().item()


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


def plot_stats_boxes(
    df,
    position,
    criterion="cr",
    category="high",
    stats_agg_func="mean",
    topx_players_presence=20,
    stats_to_show=5,
    max_players_to_show=8,
):
    infovars = ["slug", "team_name", "position", "team_code"]

    variables = [
        "fgm",
        "tpm",
        "ftm",
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

    # select player with enough minutes and in specific position
    df_position = (
        df[df.position == position]
        .loc[lambda x: x.slug.isin(x.groupby("slug")["min"].mean().nlargest(topx_players_presence).index)]
        .assign(greedy=lambda x: x.valuation / x.cr)
    )

    # split into categories - low, medium, high
    player_cuts = pd.qcut(
        df_position.groupby("slug")[criterion].mean(), q=3, labels=["low", "medium", "high"]
    ).to_dict()

    df_filtered = df_position.loc[lambda x: x.slug.isin([x for x in player_cuts if player_cuts[x] == category])]

    if df_filtered.slug.nunique() > 8:
        df_filtered = df_filtered.loc[
            lambda x: x.slug.isin(x.groupby("slug")[criterion].mean().nlargest(max_players_to_show).index)
        ]

    # select most relevant stats
    variables_filtered = df_filtered[variables].agg(stats_agg_func).nlargest(stats_to_show).index

    # melt the dataframe
    df_long = df_filtered.melt(id_vars=infovars, value_vars=variables_filtered)

    # Set the color palette
    palette = sns.color_palette("Paired")

    # Create the plot
    sns.boxplot(data=df_long, x="variable", y="value", hue="slug", palette=palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.show()


def plot_stats_lines(df, slug, stats_agg_func="mean", stats_to_show=3):
    df_player = df[df.slug == slug].copy()

    variables = [
        "fgm",
        "tpm",
        "ftm",
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

    evalvars = ["valuation", "cr"]

    # select most relevant stats
    variables_filtered = df_player[variables].agg(stats_agg_func).nlargest(stats_to_show).index

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    for var in evalvars:
        ax1.plot(df_player.week, df_player[var], label=var)
    ax1.set_title("evaluation over time")
    ax1.legend()

    for var in variables_filtered:
        ax2.plot(df_player.week, df_player[var], label=var)
    ax2.set_title("statistics over time")
    ax2.legend()

    ax1.set_xticks(df_player.week)
    ax1.set_xticklabels(df_player.week)
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()
