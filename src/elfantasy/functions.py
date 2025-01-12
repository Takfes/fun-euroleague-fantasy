import time
from operator import le

# import highspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import requests
import seaborn as sns
from category_encoders import TargetEncoder
from euroleague_api.game_stats import GameStats
from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, PowerTransformer, StandardScaler
from xgboost import XGBRegressor


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


# softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


@timeit
def build_opt_model(data, value_col, budget=100, use_solver="glpk", guards=4, forwards=4, centers=2):
    df = data.copy()
    max_players = guards + forwards + centers

    # df = dfw.copy()
    # guards = 4
    # forwards = 4
    # centers = 2
    # value_col = "valuation"
    # use_solver = "glpk"
    # budget = 100

    # prepare data -------------------------------------------------

    slugs = df["slug"].unique()
    positions = sorted(df["position"].unique())
    posg = df[df.position == "G"]["slug"].unique()
    posf = df[df.position == "F"]["slug"].unique()
    posc = df[df.position == "C"]["slug"].unique()
    values = df[["slug", value_col]].fillna(0).set_index("slug").to_dict()[value_col]
    costs = df[["slug", "cr"]].fillna(0).set_index("slug").to_dict()["cr"]

    # create possible combos for the main 5 positions
    combos = [[2, 2, 1], [1, 2, 2], [2, 1, 2], [1, 3, 1], [3, 1, 1]]
    combosdf = pd.DataFrame(combos, columns=["G", "F", "C"])[positions]
    ct = combosdf.stack().to_dict()
    ppl = {p: df[df.position == p]["slug"].unique().tolist() for p in positions}

    # create model -------------------------------------------------

    model = pyo.ConcreteModel()

    # define sets -------------------------------------------------

    model.P = pyo.Set(initialize=slugs)
    model.pos = pyo.Set(initialize=positions)
    model.posg = pyo.Set(initialize=posg)
    model.posf = pyo.Set(initialize=posf)
    model.posc = pyo.Set(initialize=posc)
    model.combs = pyo.Set(initialize=combosdf.index)

    # define parameters -------------------------------------------------

    model.v = pyo.Param(model.P, initialize=values)
    model.c = pyo.Param(model.P, initialize=costs)
    model.guards = pyo.Param(initialize=guards)
    model.forwards = pyo.Param(initialize=forwards)
    model.centers = pyo.Param(initialize=centers)
    model.budget = pyo.Param(initialize=budget)
    model.ct = pyo.Param(model.combs, model.pos, initialize=ct)
    model.ppl = pyo.Param(model.pos, initialize=ppl)

    # define decision variables -------------------------------------------------

    model.xs5 = pyo.Var(model.P, within=pyo.Binary)
    model.xb1 = pyo.Var(model.P, within=pyo.Binary)
    model.xbr = pyo.Var(model.P, within=pyo.Binary)
    model.xc = pyo.Var(model.combs, within=pyo.Binary)
    model.z = pyo.Var(model.P, model.combs, within=pyo.Binary)

    # define constraints -------------------------------------------------

    def cst_select_player_once(model, p):
        return model.xs5[p] + model.xb1[p] + model.xbr[p] <= 1

    def cstr_starting_five(model):
        return sum(model.xs5[p] for p in model.P) == 5

    def cstr_bench_first(model):
        return sum(model.xb1[p] for p in model.P) == 1

    def cstr_bench_rest(model):
        return sum(model.xbr[p] for p in model.P) == 4

    def cstr_centers(model):
        return sum(model.xs5[p] + model.xb1[p] + model.xbr[p] for p in model.posc) == model.centers

    def cstr_forwards(model):
        return sum(model.xs5[p] + model.xb1[p] + model.xbr[p] for p in model.posf) == model.forwards

    def cstr_guards(model):
        return sum(model.xs5[p] + model.xb1[p] + model.xbr[p] for p in model.posg) == model.guards

    def cstr_budget(model):
        return sum((model.xs5[p] + model.xb1[p] + model.xbr[p]) * model.c[p] for p in model.P) <= model.budget

    def cstr_single_combo(model):
        return sum(model.xc[:]) == 1

    # ! this is how the constraint should be, however this is not supported by the solver, because it is non-linear, because there is a product of two decision variables on the lhs part of the equation
    # ! instead we are going to linearize this constraint by introducing a new decision variable z[p, comb] which is binary and is equal to 1 if player p is in the combination comb

    # def cstr_enforce_combo(model, pos):
    #     lhs = sum(model.xs5[p] * model.xc[comb] for p in model.ppl[pos] for comb in model.combs)
    #     rhs = sum(model.xc[comb] * model.ct[comb, pos] for comb in model.combs)
    #     return lhs == rhs

    def cstr_enforce_combo(model, pos):
        lhs = sum(model.z[p, comb] for p in model.ppl[pos] for comb in model.combs)
        rhs = sum(model.xc[comb] * model.ct[comb, pos] for comb in model.combs)
        return lhs == rhs

    # inject constraints in the model -------------------------------------------------

    model.cst_select_player_once = pyo.Constraint(model.P, rule=cst_select_player_once)
    model.cstr_starting_five = pyo.Constraint(rule=cstr_starting_five)
    model.cstr_bench_first = pyo.Constraint(rule=cstr_bench_first)
    model.cstr_bench_rest = pyo.Constraint(rule=cstr_bench_rest)
    model.cstr_centers = pyo.Constraint(rule=cstr_centers)
    model.cstr_forwards = pyo.Constraint(rule=cstr_forwards)
    model.cstr_guards = pyo.Constraint(rule=cstr_guards)
    model.cstr_budget = pyo.Constraint(rule=cstr_budget)
    model.cstr_single_combo = pyo.Constraint(rule=cstr_single_combo)
    model.cstr_enforce_combo = pyo.Constraint(model.pos, rule=cstr_enforce_combo)

    # ! linearization of the product of two decision variables
    model.cstr_product_linearization = pyo.ConstraintList()
    for p in model.P:
        for comb in model.combs:
            model.cstr_product_linearization.add(model.z[p, comb] <= model.xs5[p])
            model.cstr_product_linearization.add(model.z[p, comb] <= model.xc[comb])
            model.cstr_product_linearization.add(model.z[p, comb] >= model.xs5[p] + model.xc[comb] - 1)

    # define objective -------------------------------------------------

    def obj(model):
        return sum((model.xs5[p] + model.xb1[p] + 0.5 * model.xbr[p]) * model.v[p] for p in model.P)

    model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    # solver options ------------------------------------------------

    if use_solver == "appsi_highs":
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
    else:
        solver = pyo.SolverFactory(use_solver)

    # solve problem -------------------------------------------------

    results = solver.solve(model, tee=False)

    # solve problem -------------------------------------------------

    if results.solver.status != pyo.SolverStatus.ok:
        raise ValueError("Check solver not ok")
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError("Check solver termination condition not optimal")

    # postprocessing -------------------------------------------------

    # extract selected combo
    selected_combo = [k for k, v in model.xc.extract_values().items() if v > 0.55][0]
    selected_combo_positions = ", ".join([f"{k}={v}" for k, v in combosdf.loc[selected_combo, :].to_dict().items()])
    selected_combo_summary = f"combo : {selected_combo} | positions : {selected_combo_positions}"

    # extract selected player details
    starting_5 = {k: int(v > 0.55) for k, v in model.xs5.extract_values().items()}
    bench_first = {k: int(v > 0.55) for k, v in model.xb1.extract_values().items()}
    bench_rest = {k: int(v > 0.55) for k, v in model.xbr.extract_values().items()}

    df["starting_5"] = df["slug"].map(starting_5)
    df["bench_first"] = df["slug"].map(bench_first)
    df["bench_rest"] = df["slug"].map(bench_rest)
    df["solution"] = df["starting_5"] + df["bench_first"] + df["bench_rest"]

    df1 = df[df.solution == 1]

    # objective value
    # objective_value = model.obj()
    objective_value = (
        (
            df1["valuation"] * df1["starting_5"]
            + df1["valuation"] * df1["bench_first"]
            + 0.5 * df1["valuation"] * df1["bench_rest"]
        )
        .sum()
        .item()
    )

    # validate solution ------------------------------------------------

    assert df1.shape[0] == max_players, f"More than {max_players} players selected"

    assert df1[df1.position == "G"].slug.nunique() == guards, f"More than {guards} guards selected"

    assert df1[df1.position == "F"].slug.nunique() == forwards, f"More than {forwards} forwards selected"

    assert df1[df1.position == "C"].slug.nunique() == centers, f"More than {centers} centers selected"

    assert df1.cr.sum().item() <= budget + 0.001, f"Budget exceeded: {df1.cr.sum().item()} > {budget}"

    assert df1.groupby("team_code").size().max().item() <= 6, "More than 6 players from the same team selected"

    # prepare solution summary ------------------------------------------------

    summary = {
        "objective_value": objective_value,
        "total_cost": df1.cr.sum().item(),
        "players": df1.slug.unique().tolist(),
        "starting_five": df1[df1.starting_5 == 1].slug.unique().tolist(),
        "bench_first": df1[df1.bench_first == 1].slug.unique().tolist(),
        "bench_rest": df1[df1.bench_rest == 1].slug.unique().tolist(),
        "centers": df1[df1.position == "C"].slug.unique().tolist(),
        "forwards": df1[df1.position == "F"].slug.unique().tolist(),
        "guards": df1[df1.position == "G"].slug.unique().tolist(),
        "selected_combo_summary": selected_combo_summary,
    }

    return summary


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


def plot_regression_diagnostics(y_test, y_pred):
    # Calculate errors
    errors = y_test - y_pred

    # Create a figure with four subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    # Plot errors vs y_test
    axes[0, 0].scatter(y_test, errors, alpha=0.5)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Errors")
    axes[0, 0].set_title("Errors vs Actual Values")

    # Plot errors vs y_pred
    axes[0, 1].scatter(y_pred, errors, alpha=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Errors")
    axes[0, 1].set_title("Errors vs Predicted Values")

    # Plot y_test vs y_pred
    axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[1, 0].set_xlabel("Actual Values")
    axes[1, 0].set_ylabel("Predicted Values")
    axes[1, 0].set_title("Actual vs Predicted Values")

    # Plot feature importances or coefficients
    axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label="Predicted")
    axes[1, 1].hist(y_test, bins=30, alpha=0.7, label="Actual")
    axes[1, 1].set_title("Predicted vs Actual Distribution")
    axes[1, 1].legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


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
        dfc[f"{feature}_expanding"] = (
            dfc.sort_values(by=["week"], ascending=True)
            .groupby(["slug"])[feature]
            .apply(lambda x: x.shift().expanding().mean())
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


def build_pred_model(model_string, nums, cats, transform_target=False):
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("feature_agglomeration", FeatureAgglomeration(n_clusters=10)),
            # ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = TargetEncoder()

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, nums),
            ("cat", categorical_transformer, cats),
        ]
    )

    # Define the models
    model_lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.001,
        num_leaves=71,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=1990,
    )

    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        min_child_weight=1,
        random_state=1990,
    )

    model_knn = KNeighborsRegressor(
        n_neighbors=10,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
    )

    model_lr = LinearRegression()

    # Add the linear regression model to the list of models
    models = {
        "LGBM": model_lgbm,
        "XGB": model_xgb,
        "KNN": model_knn,
        "LR": model_lr,
    }

    # Choose the model for hyperparameter optimization
    model = models[model_string]  # or any other model from the models dictionary

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Create a power transformer
    if transform_target:
        return TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer(method="yeo-johnson"))
    else:
        return pipeline
