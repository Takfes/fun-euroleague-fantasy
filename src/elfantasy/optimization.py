import highspy
import pandas as pd
import pyomo.environ as pyo

from elfantasy.utils import timeit


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
