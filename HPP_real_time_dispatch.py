from __future__ import annotations
from dataclasses import dataclass
import os
import time
import platform
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB




@dataclass(frozen=True)
class Gen:
    name: str
    pmin: float
    pmax: float
    noload_cost: float
    inc_cost: float
    ramp_mw_per_hour: float

@dataclass(frozen=True)
class BESS:
    name: str
    e_mwh: float
    p_ch_max: float
    p_dis_max: float
    eta_c: float
    eta_d: float
    soc_min: float
    soc_max: float

@dataclass
class LyapunovParams:
    future_lookahead: int
    beta: float
    theta_soc: float
    sigma_loss: float
    max_sigma_iters: int
    sigma_scale: float
    scd_tol: float
    reserve_reaction_time: float
    lol_penalty: float
    curtail_penalty: float
    reserve_short_penalty: float
    lineflow_penalty: float
    loss_factor: float
    netcharge_penalty: float


LOL_TOL = 1e-6
LINEFLOW_TOL = 1e-6
GEN_RESERVE_PRICE = 0.01
BESS_RESERVE_PRICE = 0.001

def get_implementation_info(dispatch_interval_hours: float, forecast_horizon_hours: float) -> pd.DataFrame:


    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        memory_text = f"{total_memory_gb:.2f} GB RAM"
    except Exception:
        memory_text = "UNKNOWN_RAM; run: pip install psutil, or check manually"

    processor = platform.processor() or platform.machine() or "UNKNOWN_PROCESSOR"

    try:
        gurobi_version = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        gurobi_version = "UNKNOWN_GUROBI_VERSION"

    return pd.DataFrame([
        {"Item": "Modeling platform", "Value": f"Python {platform.python_version()} / Python script on {platform.system()} {platform.release()}"},
        {"Item": "Optimization solver and version", "Value": f"Gurobi {gurobi_version}"},
        {"Item": "Processor and memory", "Value": f"{processor}; {memory_text}"},
        {"Item": "Dispatch interval", "Value": f"{dispatch_interval_hours:g} hour"},
        {"Item": "Forecast horizon length m", "Value": f"{forecast_horizon_hours:g} h"},
        {"Item": "Total solve time for all intervals", "Value": "FILLED_AFTER_RUN"},
        {"Item": "Average solve time per interval", "Value": "FILLED_AFTER_RUN"},
        {"Item": "Maximum solve time per interval", "Value": "FILLED_AFTER_RUN"},
    ])

def finalize_implementation_info(df_results: pd.DataFrame, dispatch_interval_hours: float, forecast_horizon_hours: float) -> pd.DataFrame:

    info = get_implementation_info(dispatch_interval_hours, forecast_horizon_hours)
    if "solve_time_seconds" not in df_results.columns or df_results.empty:
        return info

    total_s = float(df_results["solve_time_seconds"].sum())
    avg_s = float(df_results["solve_time_seconds"].mean())
    max_s = float(df_results["solve_time_seconds"].max())
    info.loc[info["Item"] == "Total solve time for all intervals", "Value"] = f"{total_s:.6f} s ({total_s / 60.0:.3f} min)"
    info.loc[info["Item"] == "Average solve time per interval", "Value"] = f"{avg_s:.6f} s"
    info.loc[info["Item"] == "Maximum solve time per interval", "Value"] = f"{max_s:.6f} s"
    return info

def norm_bus(x) -> str:

    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s

def read_timeseries_xlsx(path: str) -> np.ndarray:
    df = read_table(path)

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(float).values

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        raise ValueError(f"{path} has no numeric columns.")

    if "Hour" in df.columns and len(numeric_cols) >= 2:
        for c in numeric_cols:
            if c != "Hour":
                return df[c].astype(float).values

    return df[numeric_cols[0]].astype(float).values

def read_commitment(path: str, gens: list[Gen]) -> np.ndarray:
    df = read_table(path)
    if "Hour" in df.columns:
        df = df.sort_values("Hour").reset_index(drop=True)

    T = df.shape[0]
    u = np.zeros((T, len(gens)), dtype=int)

    for i, g in enumerate(gens):
        if g.name not in df.columns:
            raise ValueError(f"Commitment file missing generator column: {g.name}")
        values = pd.to_numeric(df[g.name], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"Commitment must be binary: {g.name}")
        u[:, i] = values.astype(int).values

    return u

def load_sf_matrix(sf_xlsx_path: str) -> pd.DataFrame:
    sf = read_table(sf_xlsx_path, index_col=0)
    sf.columns = [norm_bus(c) for c in sf.columns]

    new_index = []
    for s in sf.index.astype(str):
        parts = s.split("_")
        if len(parts) >= 3 and parts[0].startswith("L"):
            new_index.append(f"{parts[0]}_{norm_bus(parts[1])}_{norm_bus(parts[2])}")
        else:
            new_index.append(s)
    sf.index = new_index
    return sf

def load_line_limits(line_xlsx_path: str) -> pd.DataFrame:
    lf = read_table(line_xlsx_path)
    needed = {"StartBusName", "EndBusName", "X", "FlowLim"}
    if not needed.issubset(set(lf.columns)):
        raise ValueError(f"{line_xlsx_path} must contain columns {needed}, got {list(lf.columns)}")
    lf["StartBusName"] = lf["StartBusName"].apply(norm_bus)
    lf["EndBusName"] = lf["EndBusName"].apply(norm_bus)
    lf["FlowLim"] = lf["FlowLim"].astype(float)
    return lf

def build_line_index_from_linefile(linefile: pd.DataFrame) -> list[str]:
    idx = []
    for i, row in linefile.reset_index(drop=True).iterrows():
        fb = norm_bus(row["StartBusName"])
        tb = norm_bus(row["EndBusName"])
        idx.append(f"L{i+1}_{fb}_{tb}")
    return idx

def read_bus_load_timeseries(path: str) -> pd.DataFrame:
    df = read_table(path)
    if "hour" not in df.columns:
        raise ValueError(f"{path} missing 'hour' column.")
    df = df.sort_values("hour").reset_index(drop=True)

    bus_cols = [c for c in df.columns if c != "hour"]
    df.rename(columns={c: norm_bus(c) for c in bus_cols}, inplace=True)

    out = df.set_index("hour")
    out.index = out.index.astype(int)
    for c in out.columns:
        out[c] = out[c].astype(float)
    return out

def read_other_fixed_gen_xlsx(path: str) -> pd.DataFrame:
    df = read_table(path)
    if "hour" not in df.columns:
        raise ValueError(f"{path} missing 'hour' column.")
    df = df.sort_values("hour").set_index("hour")
    df.index = df.index.astype(int)
    for c in df.columns:
        df[c] = df[c].astype(float)
    return df

def precompute_sf_terms(sf_df: pd.DataFrame, eps: float = 1e-12):
    buses = list(sf_df.columns)
    terms_by_line = []
    for _, row in sf_df.iterrows():
        nz = []
        for b in buses:
            v = float(row[b])
            if abs(v) > eps:
                nz.append((b, v))
        terms_by_line.append(nz)
    return buses, terms_by_line

from pathlib import Path
from types import SimpleNamespace


def read_table(path, sheet_name=0, **kwargs):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def required_table(path, sheet, columns, allow_empty=False):
    table = read_table(path, sheet).dropna(how="all")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{sheet}: missing columns {sorted(missing)}")
    if table.empty and not allow_empty:
        raise ValueError(f"{sheet}: no input rows")
    if table[list(columns)].isna().any().any():
        raise ValueError(f"{sheet}: a required input is blank")
    return table


def check_unique(values, label):
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate identifiers in {label}")


def finite_number(value, label):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def integer_value(value, label, minimum=0):
    value = finite_number(value, label)
    if value != int(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def build_hpp_incidence(hpp, pv, bess):
    HPP_units = hpp["HPP_name"].astype(str).tolist()
    PV_units = pv["name"].astype(str).tolist()
    BESS_units = bess["name"].astype(str).tolist()
    check_unique(HPP_units, "HPP model names")
    pv_index = {name: i for i, name in enumerate(PV_units)}
    bess_index = {name: i for i, name in enumerate(BESS_units)}
    A_HPP_VRE = np.zeros((len(HPP_units), len(PV_units)))
    A_HPP_BESS = np.zeros((len(HPP_units), len(BESS_units)))
    for k, (_, row) in enumerate(hpp.iterrows()):
        for column in hpp.columns:
            if not column.startswith(("internal_gen_unit_", "internal_ESS_unit_")):
                continue
            value = row[column]
            if pd.isna(value) or str(value).strip() in ("", "0", "0.0"):
                continue
            member = str(value).strip()
            if column.startswith("internal_gen_unit_"):
                if member not in pv_index:
                    raise ValueError(f"HPP {row.HPP_name}: unknown PV member {member}")
                j = pv_index[member]
                if A_HPP_VRE[:, j].sum():
                    raise ValueError(f"PV member assigned more than once: {member}")
                A_HPP_VRE[k, j] = 1
            else:
                if member not in bess_index:
                    raise ValueError(f"HPP {row.HPP_name}: unknown BESS member {member}")
                j = bess_index[member]
                if A_HPP_BESS[:, j].sum():
                    raise ValueError(f"BESS member assigned more than once: {member}")
                A_HPP_BESS[k, j] = 1
    return HPP_units, PV_units, BESS_units, A_HPP_VRE, A_HPP_BESS


def build_bess_pv_mapping(incidence):
    hpp_names, pv_names, bess_names, hpp_pv, hpp_bess = incidence
    mapping = {}
    for bi, bname in enumerate(bess_names):
        members = np.flatnonzero(hpp_bess[:, bi])
        if len(members) != 1:
            raise ValueError(f"BESS {bname} must belong to exactly one HPP")
        hi = int(members[0])
        pv_members = np.flatnonzero(hpp_pv[hi])
        if len(pv_members) != 1:
            raise ValueError(f"HPP {hpp_names[hi]} must identify exactly one PV to associate with BESS {bname}")
        mapping[bname] = pv_names[int(pv_members[0])]
    return mapping


def load_case(input_dir):
    input_dir = Path(input_dir).resolve()
    parameter_table = required_table(input_dir / "Parameters.xlsx", "Parameters", ["parameter", "value"])
    check_unique(parameter_table["parameter"], "Parameters")
    values = dict(zip(parameter_table["parameter"], parameter_table["value"]))
    values.pop("gen_reserve_price", None)
    values.pop("bess_reserve_price", None)
    model_fields = set(LyapunovParams.__dataclass_fields__)
    extra_fields = {"reserve_pv_factor", "sf_epsilon", "progress_every", "solver_output"}
    missing = (model_fields | extra_fields) - values.keys()
    if missing:
        raise ValueError(f"Missing parameters: {sorted(missing)}")
    unknown = values.keys() - (model_fields | extra_fields)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    values = {key: finite_number(value, key) for key, value in values.items()}
    for key, lower in [("future_lookahead", 0), ("max_sigma_iters", 1), ("progress_every", 0), ("solver_output", 0)]:
        values[key] = integer_value(values[key], key, lower)
    if values["solver_output"] not in (0, 1):
        raise ValueError("solver_output must be 0 or 1")
    if values["reserve_reaction_time"] <= 0 or values["sigma_scale"] <= 0:
        raise ValueError("Reserve duration and sigma_scale must be positive")
    if values["loss_factor"] < 0 or any(values[k] < 0 for k in values if k.endswith(("penalty", "tol", "price"))):
        raise ValueError("Loss factor, penalties, tolerances and reserve prices must be nonnegative")
    if values["sf_epsilon"] < 0 or values["reserve_pv_factor"] < 0:
        raise ValueError("sf_epsilon and reserve_pv_factor must be nonnegative")
    params = LyapunovParams(**{key: values[key] for key in model_fields})

    files = required_table(input_dir / "Other.xlsx", "Other", ["key", "path"])
    files = files.loc[files["key"] != "output_dir"]
    check_unique(files["key"], "Other")
    paths = {str(r.key): (input_dir / str(r.path)).resolve()
             for r in files.itertuples(index=False)}
    required_files = {"commitment", "gsf", "loads", "lines", "other_generation"}
    if set(paths) != required_files:
        raise ValueError(f"Other must specify exactly these keys: {sorted(required_files)}")
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} file: {path}")
    paths["output_dir"] = (input_dir / "../outputs").resolve()

    gcols = ["name", "bus", "pmin", "pmax", "noload_cost", "inc_cost", "ramp_mw_per_hour", "emergency_order"]
    bcols = ["name", "bus", "e_mwh", "p_ch_max", "p_dis_max", "eta_c", "eta_d", "soc_min", "soc_max", "initial_soc"]
    pcols = ["name", "bus", "forecast_file", "realized_file", "reserve_weight"]
    generators = required_table(input_dir / "Generator.xlsx", "Generator", gcols)
    storage = required_table(input_dir / "BESS.xlsx", "BESS", bcols)
    pv = required_table(input_dir / "PV.xlsx", "PV", pcols)
    fixed = required_table(input_dir / "OtherGeneration.xlsx", "OtherGeneration", ["name", "bus"], allow_empty=True)
    hpp = required_table(input_dir / "HPP.xlsx", "HPP", ["HPP_name", "HPP_capacity"])
    if not any(str(c).startswith("internal_gen_unit_") for c in hpp) or not any(str(c).startswith("internal_ESS_unit_") for c in hpp):
        raise ValueError("HPP requires internal_gen_unit_* and internal_ESS_unit_* columns")
    for table, label in [(generators, "Generator"), (storage, "BESS"), (pv, "PV"), (fixed, "OtherGeneration")]:
        table["name"] = table["name"].astype(str).str.strip()
        check_unique(table["name"], label)
        table["bus"] = table["bus"].apply(norm_bus)
    check_unique(hpp["HPP_name"].astype(str), "HPP names")
    for table, cols, label in [(generators, gcols[2:], "Generator"), (storage, bcols[2:], "BESS"), (pv, ["reserve_weight"], "PV"), (hpp, ["HPP_capacity"], "HPP")]:
        for col in cols:
            table[col] = [finite_number(x, f"{label}.{col}") for x in table[col]]
    gens = [Gen(r.name, r.pmin, r.pmax, r.noload_cost, r.inc_cost, r.ramp_mw_per_hour)
            for r in generators.itertuples(index=False)]
    bess = [BESS(r.name, r.e_mwh, r.p_ch_max, r.p_dis_max, r.eta_c, r.eta_d, r.soc_min, r.soc_max)
            for r in storage.itertuples(index=False)]
    for g in gens:
        if not (0 <= g.pmin <= g.pmax and g.ramp_mw_per_hour >= 0):
            raise ValueError(f"Invalid generator limits: {g.name}")
    for r in storage.itertuples(index=False):
        if not (r.e_mwh > 0 and r.p_ch_max >= 0 and r.p_dis_max >= 0 and 0 < r.eta_c <= 1 and 0 < r.eta_d <= 1 and 0 <= r.soc_min <= r.initial_soc <= r.soc_max <= 1):
            raise ValueError(f"Invalid BESS parameters: {r.name}")
    if (hpp["HPP_capacity"] <= 0).any() or (pv["reserve_weight"] < 0).any():
        raise ValueError("HPP capacities must be positive; reserve weights must be nonnegative")
    generators["emergency_order"] = [integer_value(x, "emergency_order") for x in generators["emergency_order"]]
    emergency = generators[generators["emergency_order"] > 0].sort_values("emergency_order")
    check_unique(emergency["emergency_order"], "emergency_order")
    incidence = build_hpp_incidence(hpp, pv, storage)
    bess_pv = build_bess_pv_mapping(incidence)
    forecast = {}
    realized = {}
    for r in pv.itertuples(index=False):
        forecast[r.name] = read_timeseries_xlsx(input_dir / str(r.forecast_file))
        realized[r.name] = read_timeseries_xlsx(input_dir / str(r.realized_file))
    case = SimpleNamespace(
        params=params, paths=paths, gens=gens, bess=bess, incidence=incidence,
        gen_bus=dict(zip(generators["name"], generators["bus"])),
        bess_bus=dict(zip(storage["name"], storage["bus"])),
        pv_bus=dict(zip(pv["name"], pv["bus"])),
        bess_pv=bess_pv,
        fixed_map=dict(zip(fixed["name"], fixed["bus"])),
        initial_soc=dict(zip(storage["name"], storage["initial_soc"])),
        hpp_capacity=dict(zip(incidence[0], hpp["HPP_capacity"])),
        emergency_order=emergency["name"].tolist(),
        forecast=forecast, realized=realized,
        reserve_weights={str(r.name): float(r.reserve_weight) for r in pv.itertuples(index=False) if r.reserve_weight != 0},
        reserve_pv_factor=values["reserve_pv_factor"], sf_epsilon=values["sf_epsilon"],
        progress_every=values["progress_every"], solver_output=values["solver_output"],
    )
    return case


def validate_timeseries(case, u_all, loadbus_df, other_df, sf_buses):
    T = len(u_all)
    if T == 0:
        raise ValueError("Input series are empty")
    if any(len(x) != T for x in list(case.forecast.values()) + list(case.realized.values())):
        raise ValueError("Commitment and all PV series must have equal lengths")
    expected = list(range(1, T + 1))
    for label, df in [("loads", loadbus_df), ("fixed generation", other_df)]:
        if df.index.tolist() != expected:
            raise ValueError(f"{label}: hour values must be unique and cover 1 through {T}")
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: non-finite values")
    for label, series in list(case.forecast.items()) + list(case.realized.items()):
        if not np.isfinite(series).all():
            raise ValueError(f"{label}: non-finite values")
    for name, series in list(case.forecast.items()) + list(case.realized.items()):
        if (series < 0).any():
            raise ValueError(f"PV values must be nonnegative: {name}")
    buses = set(sf_buses)
    mapped = set(case.gen_bus.values()) | set(case.bess_bus.values()) | set(case.pv_bus.values()) | set(case.fixed_map.values()) | set(loadbus_df.columns)
    if mapped - buses:
        raise ValueError(f"Nodes missing from GSF columns: {sorted(mapped - buses)}")
    if set(case.fixed_map) - set(other_df.columns):
        raise ValueError("A configured fixed generator is missing from its time series")

def solve_horizon_once(
    t: int,
    H: int,
    gens: list[Gen],
    bess_list: list[BESS],
    u_window: np.ndarray,
    loadbus_rows: list[dict[str, float]],
    fixed_bus_inj_rows: list[dict[str, float]],
    pv_dict_rows: list[dict[str, float]],
    soc_prev: dict[str, float],
    J_prev: dict[str, float],
    reg_up_req_rows: list[float],
    P_prev: dict[str, float],
    params: LyapunovParams,
    sigma_loss: float,
    HPP_capacity_limit: dict[str, float],
    enable_line_constraints: bool,
    sf_buses: list[str] | None,
    sf_terms_by_line: list[list[tuple[str, float]]] | None,
    flow_limits: np.ndarray | None,
    line_names: list[str] | None,
    HPP_units,
    PV_units,
    BESS_units,
    A_HPP_VRE,
    A_HPP_BESS,
    case,
):


    m = gp.Model(f"forecast_enhanced_hour_{t}_H{H}")
    m.Params.OutputFlag = case.solver_output

    gen_bus = case.gen_bus
    bess_bus = case.bess_bus
    pv_bus = case.pv_bus


    P = {(tau, g.name): m.addVar(lb=0, name=f"P_{tau}_{g.name}") for tau in range(H) for g in gens}
    Rup_g = {(tau, g.name): m.addVar(lb=0, name=f"RupG_{tau}_{g.name}") for tau in range(H) for g in gens}

    pplus = {(tau, b.name): m.addVar(lb=0, ub=b.p_dis_max, name=f"Pdis_{tau}_{b.name}") for tau in range(H) for b in bess_list}
    pminus = {(tau, b.name): m.addVar(lb=0, ub=b.p_ch_max, name=f"Pch_{tau}_{b.name}") for tau in range(H) for b in bess_list}
    Rup_b = {(tau, b.name): m.addVar(lb=0, name=f"RupB_{tau}_{b.name}") for tau in range(H) for b in bess_list}
    SOC = {(tau, b.name): m.addVar(lb=b.soc_min, ub=b.soc_max, name=f"SOC_{tau}_{b.name}") for tau in range(H) for b in bess_list}

    pnet_ch = {(tau, h): m.addVar(lb=0, name=f"PnetCh_{tau}_{h}") for tau in range(H) for h in HPP_units}

    LOL = {tau: m.addVar(lb=0, name=f"LOL_{tau}") for tau in range(H)}
    Rshort = {tau: m.addVar(lb=0, name=f"Reserve_shortfall_{tau}") for tau in range(H)}

    Curt = {(tau, p): m.addVar(lb=0, name=f"Curt_{tau}_{p}") for tau in range(H) for p in PV_units}


    SLp = {}
    SLn = {}
    if enable_line_constraints:
        assert sf_buses is not None and sf_terms_by_line is not None and flow_limits is not None
        n_line = len(flow_limits)
        for tau in range(H):
            for li in range(n_line):
                SLp[(tau, li)] = m.addVar(lb=0, name=f"SLp_{tau}_{li+1}")
                SLn[(tau, li)] = m.addVar(lb=0, name=f"SLn_{tau}_{li+1}")


    for tau in range(H):
        u_tau = u_window[tau]
        loadbus_row = loadbus_rows[tau]
        fixed_bus_inj = fixed_bus_inj_rows[tau]
        pv_dict = pv_dict_rows[tau]


        for i, g in enumerate(gens):
            if u_tau[i] == 0:
                m.addConstr(P[(tau, g.name)] == 0, name=f"offP_{tau}_{g.name}")
                m.addConstr(Rup_g[(tau, g.name)] == 0, name=f"offR_{tau}_{g.name}")
            else:
                m.addConstr(P[(tau, g.name)] >= g.pmin, name=f"pmin_{tau}_{g.name}")
                m.addConstr(P[(tau, g.name)] <= g.pmax, name=f"pmax_{tau}_{g.name}")
                m.addConstr(P[(tau, g.name)] + Rup_g[(tau, g.name)] <= g.pmax, name=f"headroom_{tau}_{g.name}")

            if tau == 0:
                p_start = P_prev[g.name]
            else:
                p_start = P[(tau - 1, g.name)]


            m.addConstr(
                P[(tau, g.name)] - p_start + Rup_g[(tau, g.name)] <= g.ramp_mw_per_hour,
                name=f"ramp_up_with_reserve_{tau}_{g.name}",
            )
            m.addConstr(p_start - P[(tau, g.name)] <= g.ramp_mw_per_hour, name=f"ramp_dn_{tau}_{g.name}")

            gen_rup_ramp_cap = g.ramp_mw_per_hour * params.reserve_reaction_time / 60.0
            m.addConstr(
                Rup_g[(tau, g.name)] <= gen_rup_ramp_cap,
                name=f"gen_rup_ramp_cap_{tau}_{g.name}",
            )


        for b in bess_list:
            pv_name = case.bess_pv[b.name]
            pv_b = float(pv_dict[pv_name])
            soc_start = soc_prev[b.name] if tau == 0 else SOC[(tau - 1, b.name)]

            m.addConstr(
                SOC[(tau, b.name)] == soc_start
                + (b.eta_c * pminus[(tau, b.name)] - (1 / b.eta_d) * pplus[(tau, b.name)]) / b.e_mwh,
                name=f"soc_dyn_{tau}_{b.name}",
            )

            m.addConstr(Curt[(tau, pv_name)] <= pv_b, name=f"curt_ub_{tau}_{b.name}")

            m.addConstr(
                Rup_b[(tau, b.name)] <= (b.p_dis_max - pplus[(tau, b.name)]) + pminus[(tau, b.name)],
                name=f"bess_rup_cap_{tau}_{b.name}",
            )


            reserve_duration_h = params.reserve_reaction_time / 60.0

            m.addConstr(
                Rup_b[(tau, b.name)] <=
                (SOC[(tau, b.name)] - b.soc_min) * b.e_mwh * b.eta_d / reserve_duration_h,
                name=f"bess_rup_cap_soc_{tau}_{b.name}",
            )


        for p in PV_units:
            if p not in case.bess_pv.values() and p in pv_dict:
                m.addConstr(Curt[(tau, p)] <= float(pv_dict[p]), name=f"curt_ub_{tau}_{p}")


        for h_idx, h in enumerate(HPP_units):
            pv_part = gp.LinExpr()
            for p_idx, p in enumerate(PV_units):
                if p in pv_dict:
                    pv_val = float(pv_dict[p])
                    if (tau, p) in Curt:
                        pv_part += float(A_HPP_VRE[h_idx, p_idx]) * (pv_val - Curt[(tau, p)])
                    else:
                        pv_part += float(A_HPP_VRE[h_idx, p_idx]) * pv_val

            bess_part = gp.LinExpr()
            for b_idx, bname in enumerate(BESS_units):
                bess_part += float(A_HPP_BESS[h_idx, b_idx]) * (
                    pplus[(tau, bname)] - pminus[(tau, bname)]
                )

            m.addConstr(pv_part + bess_part + pnet_ch[(tau, h)] >= 0, name=f"hpp_netch_{tau}_{h}")


        for h_idx, h in enumerate(HPP_units):
            bess_part_up = gp.LinExpr()
            bess_part_dn = gp.LinExpr()

            for b_idx, bname in enumerate(BESS_units):
                bess_part_up += float(A_HPP_BESS[h_idx, b_idx]) * (
                    pplus[(tau, bname)] - pminus[(tau, bname)] + Rup_b[(tau, bname)]
                )
                bess_part_dn += float(A_HPP_BESS[h_idx, b_idx]) * (
                    pplus[(tau, bname)] - pminus[(tau, bname)]
                )

            pv_part = gp.LinExpr()
            for p_idx, p in enumerate(PV_units):
                if p in pv_dict:
                    pv_part += float(A_HPP_VRE[h_idx, p_idx]) * (
                        float(pv_dict[p]) - Curt[(tau, p)]
                    )

            limit = HPP_capacity_limit[h]
            m.addConstr(bess_part_up + pv_part <= limit, name=f"hpp_cap_up_{tau}_{h}")
            m.addConstr(bess_part_dn + pv_part >= -limit, name=f"hpp_cap_dn_{tau}_{h}")


        total_load_adj = (1.0 + params.loss_factor) * float(sum(loadbus_row.values()))
        total_fixed = float(sum(fixed_bus_inj.values()))

        m.addConstr(
            gp.quicksum(P[(tau, g.name)] for g in gens)
            + gp.quicksum(pplus[(tau, b.name)] - pminus[(tau, b.name)] for b in bess_list)
            + gp.quicksum(float(pv_dict[p]) - Curt[(tau, p)] for p in PV_units if p in pv_dict)
            + total_fixed
            + LOL[tau]
            == total_load_adj,
            name=f"power_balance_consistent_{tau}",
        )


        m.addConstr(
            gp.quicksum(Rup_g[(tau, g.name)] for g in gens)
            + gp.quicksum(Rup_b[(tau, b.name)] for b in bess_list)
            + Rshort[tau]
            >= reg_up_req_rows[tau],
            name=f"reg_up_req_{tau}",
        )


        if enable_line_constraints:
            assert sf_buses is not None and sf_terms_by_line is not None and flow_limits is not None
            p_inj = {bus: gp.LinExpr() for bus in sf_buses}


            for bus in sf_buses:
                ld_raw = float(loadbus_row.get(bus, 0.0))
                ld_adj = (1.0 + params.loss_factor) * ld_raw
                if abs(ld_adj) > 0:
                    p_inj[bus].addConstant(-ld_adj)


            for bus, val in fixed_bus_inj.items():
                bn = norm_bus(bus)
                if bn in p_inj and abs(val) > 0:
                    p_inj[bn].addConstant(float(val))


            for g in gens:
                bus = gen_bus.get(g.name, None)
                if bus is not None and bus in p_inj:
                    p_inj[bus] += P[(tau, g.name)]


            for b in bess_list:
                bus = bess_bus.get(b.name, None)
                if bus is not None and bus in p_inj:
                    p_inj[bus] += pplus[(tau, b.name)] - pminus[(tau, b.name)]


            for p in PV_units:
                bus = pv_bus.get(p, None)
                if bus is not None and bus in p_inj and p in pv_dict:
                    p_inj[bus] += float(pv_dict[p]) - Curt[(tau, p)]

            for li in range(len(flow_limits)):
                flow_expr = gp.LinExpr()
                for (bus, coeff) in sf_terms_by_line[li]:
                    flow_expr += float(coeff) * p_inj[bus]
                fmax = float(flow_limits[li])

                m.addConstr(flow_expr <= fmax + SLp[(tau, li)], name=f"line_ub_{tau}_{li+1}")
                m.addConstr(-flow_expr <= fmax + SLn[(tau, li)], name=f"line_lb_{tau}_{li+1}")


    totalE = sum(b.e_mwh for b in bess_list)
    obj = gp.LinExpr()

    for tau in range(H):
        u_tau = u_window[tau]
        pv_dict = pv_dict_rows[tau]

        gen_cost = gp.quicksum(
            (g.inc_cost * (P[(tau, g.name)] - g.pmin) + g.noload_cost * g.pmin) if u_tau[i] == 1 else 0
            for i, g in enumerate(gens)
        )

        reserve_cost = (
            GEN_RESERVE_PRICE * gp.quicksum(Rup_g[(tau, g.name)] for g in gens)
            + BESS_RESERVE_PRICE * gp.quicksum(Rup_b[(tau, b.name)] for b in bess_list)
        )


        drift_tau = gp.quicksum(
            params.beta * J_prev[b.name]
            * (b.eta_c * pminus[(tau, b.name)] - (1 / b.eta_d) * pplus[(tau, b.name)])
            for b in bess_list
        )

        loss_penalty = gp.quicksum(
            sigma_loss * ((1 / b.eta_d - 1) * pplus[(tau, b.name)] + (1 - b.eta_c) * pminus[(tau, b.name)])
            for b in bess_list
        )

        lol_cost = params.lol_penalty * LOL[tau]
        curt_cost = params.curtail_penalty * gp.quicksum(Curt[(tau, p)] for p in PV_units if p in pv_dict)
        reserve_short_cost = params.reserve_short_penalty * Rshort[tau]

        if enable_line_constraints:
            line_slack_cost = params.lineflow_penalty * gp.quicksum(SLp[(tau, li)] + SLn[(tau, li)] for li in range(len(flow_limits)))
        else:
            line_slack_cost = 0.0

        netcharge_cost = params.netcharge_penalty * gp.quicksum(pnet_ch[(tau, h)] for h in HPP_units)

        cost_tau = (
            gen_cost
            + reserve_cost
            + loss_penalty
            + lol_cost
            + curt_cost
            + reserve_short_cost
            + line_slack_cost
            + netcharge_cost
        )

        obj += drift_tau + cost_tau

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        m.computeIIS()
        case.paths["output_dir"].mkdir(parents=True, exist_ok=True)
        iis_name = str(case.paths["output_dir"] / f"iis_forecast_hour_{t}.ilp")
        m.write(iis_name)
        raise RuntimeError(f"Hour {t} horizon infeasible or not optimal (status={m.Status}). IIS written to {iis_name}")


    tau0 = 0
    pv_dict0 = pv_dict_rows[tau0]
    loadbus_row0 = loadbus_rows[tau0]
    fixed_bus_inj0 = fixed_bus_inj_rows[tau0]

    pv_total_t = float(sum(float(pv_dict0[p]) for p in PV_units if p in pv_dict0))
    curt_total_t = float(sum(Curt[(tau0, p)].X for p in PV_units if p in pv_dict0))
    pv_net_t = float(sum(float(pv_dict0[p]) - Curt[(tau0, p)].X for p in PV_units if p in pv_dict0))

    out = {
        "hour": t,
        "lookahead_used": H,
        "total_load_adj": float((1.0 + params.loss_factor) * sum(loadbus_row0.values())),
        "total_fixed_gen": float(sum(fixed_bus_inj0.values())),
        "pv_total": pv_total_t,
        "pv_net": pv_net_t,

        "curt_total": curt_total_t,
        "reg_up_req": float(reg_up_req_rows[tau0]),
        "reserve_shortfall": float(Rshort[tau0].X),
        "sigma_loss_used": float(sigma_loss),
        "LOL": float(LOL[tau0].X),
    }

    if enable_line_constraints:
        sl_vals = [SLp[(tau0, li)].X for li in range(len(flow_limits))] + [SLn[(tau0, li)].X for li in range(len(flow_limits))]
        out["LineSlack_sum"] = float(sum(sl_vals))
        out["LineSlack_max"] = float(max(sl_vals) if len(sl_vals) else 0.0)
    else:
        out["LineSlack_sum"] = 0.0
        out["LineSlack_max"] = 0.0

    for g in gens:
        out[f"P_{g.name}"] = float(P[(tau0, g.name)].X)
        out[f"Rup_{g.name}"] = float(Rup_g[(tau0, g.name)].X)

    for b in bess_list:
        out[f"pplus_{b.name}"] = float(pplus[(tau0, b.name)].X)
        out[f"pminus_{b.name}"] = float(pminus[(tau0, b.name)].X)
        out[f"pnet_{b.name}"] = float(pplus[(tau0, b.name)].X - pminus[(tau0, b.name)].X)
        out[f"SOC_{b.name}"] = float(SOC[(tau0, b.name)].X)
        out[f"Rup_{b.name}"] = float(Rup_b[(tau0, b.name)].X)
        out[f"PV_{case.bess_pv[b.name]}"] = float(pv_dict0[case.bess_pv[b.name]])
        out[f"Curt_{case.bess_pv[b.name]}"] = float(Curt[(tau0, case.bess_pv[b.name])].X)
        out[f"PV_net_{case.bess_pv[b.name]}"] = float(pv_dict0[case.bess_pv[b.name]] - Curt[(tau0, case.bess_pv[b.name])].X)

    for p in PV_units:
        if p not in case.bess_pv.values() and p in pv_dict0:
            out[f"PV_{p}"] = float(pv_dict0[p])
            out[f"Curt_{p}"] = float(Curt[(tau0, p)].X)
            out[f"PV_net_{p}"] = float(pv_dict0[p] - Curt[(tau0, p)].X)

    out["SOC_total"] = float(sum(SOC[(tau0, b.name)].X * b.e_mwh for b in bess_list) / totalE)
    out["BESS_net_total"] = float(sum(pplus[(tau0, b.name)].X - pminus[(tau0, b.name)].X for b in bess_list))
    out["BESS_Rup_total"] = float(sum(Rup_b[(tau0, b.name)].X for b in bess_list))
    out["GEN_total"] = float(sum(P[(tau0, g.name)].X for g in gens))
    out["GEN_Rup_total"] = float(sum(Rup_g[(tau0, g.name)].X for g in gens))


    line_violations = []
    if enable_line_constraints:
        assert sf_buses is not None and sf_terms_by_line is not None and flow_limits is not None
        p_val = {bus: 0.0 for bus in sf_buses}

        for bus in sf_buses:
            ld_raw = float(loadbus_row0.get(bus, 0.0))
            p_val[bus] -= (1.0 + params.loss_factor) * ld_raw

        for bus, val in fixed_bus_inj0.items():
            bn = norm_bus(bus)
            if bn in p_val:
                p_val[bn] += float(val)

        for g in gens:
            bus = gen_bus.get(g.name, None)
            if bus is not None and bus in p_val:
                p_val[bus] += float(P[(tau0, g.name)].X)

        for b in bess_list:
            bus = bess_bus.get(b.name, None)
            if bus is not None and bus in p_val:
                p_val[bus] += float(pplus[(tau0, b.name)].X) - float(pminus[(tau0, b.name)].X)

        for p in PV_units:
            bus = pv_bus.get(p, None)
            if bus is not None and bus in p_val and p in pv_dict0:
                p_val[bus] += float(pv_dict0[p]) - float(Curt[(tau0, p)].X)

        for li in range(len(flow_limits)):
            flow = 0.0
            for (bus, coeff) in sf_terms_by_line[li]:
                flow += float(coeff) * p_val[bus]
            fmax = float(flow_limits[li])
            viol = abs(flow) - fmax
            if viol > LINEFLOW_TOL:
                lname = line_names[li] if line_names is not None and li < len(line_names) else f"Line_{li+1}"
                direction = "pos" if flow >= 0 else "neg"
                line_violations.append({
                    "hour": t,
                    "line_index": li + 1,
                    "line_name": lname,
                    "flow_MW": flow,
                    "direction": direction,
                    "limit_MW": fmax,
                    "violation_MW": viol,
                })

    out["line_violations_detail"] = line_violations


    soc_next = {b.name: float(SOC[(tau0, b.name)].X) for b in bess_list}
    J_next = {b.name: b.e_mwh * soc_next[b.name] - params.theta_soc * b.e_mwh for b in bess_list}
    P_next = {g.name: float(P[(tau0, g.name)].X) for g in gens}


    scd_list = []
    for tau in range(H):
        for b in bess_list:
            pch = float(pminus[(tau, b.name)].X)
            pdis = float(pplus[(tau, b.name)].X)
            if pch > params.scd_tol and pdis > params.scd_tol:
                scd_list.append((tau, b.name, pch, pdis))

    return out, soc_next, J_next, P_next, scd_list

def solve_horizon_adaptive_sigma(
    t,
    H,
    gens,
    bess_list,
    u_window,
    loadbus_rows,
    fixed_bus_inj_rows,
    pv_dict_rows,
    soc_prev,
    J_prev,
    reg_up_req_rows,
    P_prev,
    params: LyapunovParams,
    HPP_capacity_limit,
    enable_line_constraints,
    sf_buses,
    sf_terms_by_line,
    flow_limits,
    line_names,
    HPP_units,
    PV_units,
    BESS_units,
    A_HPP_VRE,
    A_HPP_BESS,
    case,
):
    sigma = params.sigma_loss
    last = None

    for _ in range(params.max_sigma_iters):
        out, soc_next, J_next, P_next, scd_list = solve_horizon_once(
            t=t,
            H=H,
            gens=gens,
            bess_list=bess_list,
            u_window=u_window,
            loadbus_rows=loadbus_rows,
            fixed_bus_inj_rows=fixed_bus_inj_rows,
            pv_dict_rows=pv_dict_rows,
            soc_prev=soc_prev,
            J_prev=J_prev,
            reg_up_req_rows=reg_up_req_rows,
            P_prev=P_prev,
            params=params,
            sigma_loss=sigma,
            HPP_capacity_limit=HPP_capacity_limit,
            enable_line_constraints=enable_line_constraints,
            sf_buses=sf_buses,
            sf_terms_by_line=sf_terms_by_line,
            flow_limits=flow_limits,
            line_names=line_names,
            HPP_units=HPP_units,
            PV_units=PV_units,
            BESS_units=BESS_units,
            A_HPP_VRE=A_HPP_VRE,
            A_HPP_BESS=A_HPP_BESS,
            case=case,
        )
        last = (out, soc_next, J_next, P_next, scd_list)

        if len(scd_list) == 0:
            return out, soc_next, J_next, P_next, scd_list

        sigma *= params.sigma_scale

    assert last is not None
    return last

def solve_horizon_with_emergency_commit(
    t,
    H,
    gens,
    bess_list,
    u_window,
    loadbus_rows,
    fixed_bus_inj_rows,
    pv_dict_rows,
    soc_prev,
    J_prev,
    reg_up_req_rows,
    P_prev,
    params: LyapunovParams,
    HPP_capacity_limit,
    enable_line_constraints,
    sf_buses,
    sf_terms_by_line,
    flow_limits,
    line_names,
    HPP_units,
    PV_units,
    BESS_units,
    A_HPP_VRE,
    A_HPP_BESS,
    case,
    flex_order: list[str] | None = None,
):


    flex_order = list(case.emergency_order if flex_order is None else flex_order)
    gen_name_to_idx = {g.name: i for i, g in enumerate(gens)}

    missing = [gname for gname in flex_order if gname not in gen_name_to_idx]
    if missing:
        raise ValueError(f"Emergency flexible unit(s) not found in gens: {missing}")


    u_working = np.asarray(u_window, dtype=int).copy()
    initial_tau0_commitment = u_working[0, :].copy()
    already_on_units = [gname for gname in flex_order if int(initial_tau0_commitment[gen_name_to_idx[gname]]) == 1]

    def _solve_current_commitment():
        return solve_horizon_adaptive_sigma(
            t=t,
            H=H,
            gens=gens,
            bess_list=bess_list,
            u_window=u_working,
            loadbus_rows=loadbus_rows,
            fixed_bus_inj_rows=fixed_bus_inj_rows,
            pv_dict_rows=pv_dict_rows,
            soc_prev=soc_prev,
            J_prev=J_prev,
            reg_up_req_rows=reg_up_req_rows,
            P_prev=P_prev,
            params=params,
            HPP_capacity_limit=HPP_capacity_limit,
            enable_line_constraints=enable_line_constraints,
            sf_buses=sf_buses,
            sf_terms_by_line=sf_terms_by_line,
            flow_limits=flow_limits,
            line_names=line_names,
            HPP_units=HPP_units,
            PV_units=PV_units,
            BESS_units=BESS_units,
            A_HPP_VRE=A_HPP_VRE,
            A_HPP_BESS=A_HPP_BESS,
            case=case,
        )

    out, soc_next, J_next, P_next, scd_list = _solve_current_commitment()
    lol_before = float(out.get("LOL", 0.0))

    emergency_committed_units: list[str] = []
    emergency_attempt_log: list[str] = []
    triggered = lol_before > LOL_TOL

    if triggered:
        for gname in flex_order:
            gi = gen_name_to_idx[gname]


            if int(u_working[0, gi]) == 1:
                continue


            u_working[0, gi] = 1
            emergency_committed_units.append(gname)

            out, soc_next, J_next, P_next, scd_list = _solve_current_commitment()
            lol_now = float(out.get("LOL", 0.0))
            emergency_attempt_log.append(f"{gname}:LOL={lol_now:.9g}")

            if lol_now <= LOL_TOL:
                break

    lol_after = float(out.get("LOL", 0.0))

    out["LOL_before_emergency"] = lol_before
    out["LOL_after_emergency"] = lol_after
    out["emergency_commit_triggered"] = int(triggered)
    out["emergency_commit_count"] = int(len(emergency_committed_units))
    out["emergency_committed_units"] = ",".join(emergency_committed_units)
    out["emergency_already_on_units"] = ",".join(already_on_units)
    out["emergency_candidate_order"] = ",".join(flex_order)
    out["emergency_attempt_log"] = ";".join(emergency_attempt_log)
    out["emergency_unresolved_after_all"] = int(triggered and lol_after > LOL_TOL)

    executed_tau0_commitment = u_working[0, :].copy()
    return out, soc_next, J_next, P_next, scd_list, executed_tau0_commitment, u_working

def run_dispatch(case):
    params = case.params
    gens, bess = case.gens, case.bess
    HPP_units, PV_units, BESS_units, A_HPP_VRE, A_HPP_BESS = case.incidence
    HPP_capacity_limit = case.hpp_capacity
    u_all = read_commitment(case.paths["commitment"], gens)
    sf_path = case.paths["gsf"]
    lineparam_path = case.paths["lines"]
    loadbus_path = case.paths["loads"]
    other_fixed_path = case.paths["other_generation"]
    T = len(u_all)
    SF_df = load_sf_matrix(sf_path)
    linefile = load_line_limits(lineparam_path)
    line_names = build_line_index_from_linefile(linefile)


    SF_df = SF_df.reindex(line_names)
    if SF_df.isnull().values.any():
        missing = SF_df[SF_df.isnull().any(axis=1)].index.tolist()[:20]
        raise ValueError(
            "SF matrix missing lines after reindex. "
            f"Examples: {missing}\n"
            "Check SF row names vs line_parameters order/labels."
        )

    flow_limits = linefile["FlowLim"].astype(float).values
    sf_buses, sf_terms_by_line = precompute_sf_terms(SF_df, eps=case.sf_epsilon)

    loadbus_df = read_bus_load_timeseries(loadbus_path)
    other_df = read_other_fixed_gen_xlsx(other_fixed_path)


    fixed_map = case.fixed_map
    validate_timeseries(case, u_all, loadbus_df, other_df, sf_buses)

    enable_line_constraints = (params.lineflow_penalty is not None and params.lineflow_penalty > 0)


    soc_prev = dict(case.initial_soc)
    J_prev = {b.name: b.e_mwh * soc_prev[b.name] - params.theta_soc * b.e_mwh for b in bess}
    P_prev = {g.name: 0.0 for g in gens}

    records = []
    lol_logs = []
    scd_events = []
    all_line_violations = []
    executed_commitment_records = []

    for t in range(1, T + 1):

        H = min(params.future_lookahead + 1, T - t + 1)

        u_window = []
        loadbus_rows = []
        fixed_bus_inj_rows = []
        pv_dict_rows = []
        reg_up_req_rows = []

        for tau in range(H):
            tt = t + tau

            u_window.append(u_all[tt - 1])


            pv_series = case.realized if tau == 0 else case.forecast
            pv_dict_rows.append({p: float(pv_series[p][tt - 1]) for p in PV_units})

            if tt not in loadbus_df.index:
                raise ValueError(f"Bus load table missing hour={tt}. Check 'hour' column values.")
            rowL = loadbus_df.loc[tt]
            loadbus_rows.append({norm_bus(b): float(rowL[norm_bus(b)]) for b in loadbus_df.columns})

            if tt not in other_df.index:
                raise ValueError(f"Other fixed generation file missing hour={tt}.")
            rowF = other_df.loc[tt]
            fixed_bus_inj_t = {}
            for res_name, bus in fixed_map.items():
                if res_name not in other_df.columns:
                    raise ValueError(f"Other fixed generation file missing column {res_name}")
                fixed_bus_inj_t[norm_bus(bus)] = float(rowF[res_name])
            fixed_bus_inj_rows.append(fixed_bus_inj_t)


            reg_up_req_tau = case.reserve_pv_factor * sum(
                float(pv_dict_rows[-1][p]) * weight for p, weight in case.reserve_weights.items()
            )
            reg_up_req_rows.append(float(reg_up_req_tau))

        u_window = np.asarray(u_window, dtype=int)

        interval_solve_start = time.perf_counter()
        out, soc_next, J_next, P_next, scd_list, u_executed_tau0, u_window_after_emergency = solve_horizon_with_emergency_commit(
            t=t,
            H=H,
            gens=gens,
            bess_list=bess,
            u_window=u_window,
            loadbus_rows=loadbus_rows,
            fixed_bus_inj_rows=fixed_bus_inj_rows,
            pv_dict_rows=pv_dict_rows,
            soc_prev=soc_prev,
            J_prev=J_prev,
            reg_up_req_rows=reg_up_req_rows,
            P_prev=P_prev,
            params=params,
            HPP_capacity_limit=HPP_capacity_limit,
            enable_line_constraints=enable_line_constraints,
            sf_buses=sf_buses,
            sf_terms_by_line=sf_terms_by_line,
            flow_limits=flow_limits,
            line_names=line_names,
            HPP_units=HPP_units,
            PV_units=PV_units,
            BESS_units=BESS_units,
            A_HPP_VRE=A_HPP_VRE,
            A_HPP_BESS=A_HPP_BESS,
            case=case,
        )
        out["solve_time_seconds"] = float(time.perf_counter() - interval_solve_start)

        executed_commitment_records.append({
            "hour": t,
            **{f"u_{g.name}": int(u_executed_tau0[i]) for i, g in enumerate(gens)},
            "emergency_commit_triggered": int(out.get("emergency_commit_triggered", 0)),
            "emergency_commit_count": int(out.get("emergency_commit_count", 0)),
            "emergency_committed_units": out.get("emergency_committed_units", ""),
            "emergency_already_on_units": out.get("emergency_already_on_units", ""),
            "LOL_before_emergency": float(out.get("LOL_before_emergency", out.get("LOL", 0.0))),
            "LOL_after_emergency": float(out.get("LOL_after_emergency", out.get("LOL", 0.0))),
        })

        records.append({k: v for k, v in out.items() if k != "line_violations_detail"})

        for v in out.get("line_violations_detail", []):
            all_line_violations.append(v)

        if out["LOL"] > LOL_TOL:
            lol_logs.append({
                "hour": t,
                "lol": float(out["LOL"]),
                "soc_total": float(out["SOC_total"]),
                "lookahead_used": int(out["lookahead_used"]),
            })

        if scd_list:
            for (tau_scd, bname, pch, pdis) in scd_list:
                scd_events.append({
                    "hour": t,
                    "tau": tau_scd,
                    "absolute_hour": t + tau_scd,
                    "bess": bname,
                    "pcharge": pch,
                    "pdischarge": pdis,
                    "sigma_loss_used": out.get("sigma_loss_used", np.nan),
                    "LOL_tau0": out.get("LOL", np.nan),
                    "SOC_total_tau0": out.get("SOC_total", np.nan),


                    "PV_tau0": out.get(f"PV_{case.bess_pv[bname]}", np.nan),
                    "Curt_tau0": out.get(f"Curt_{case.bess_pv[bname]}", np.nan),
                    "PV_net_tau0": out.get(f"PV_net_{case.bess_pv[bname]}", np.nan),
                })


        soc_prev = soc_next
        J_prev = J_next
        P_prev = P_next
        if case.progress_every and (t == 1 or t % case.progress_every == 0 or t == T):
            print(f"Completed {t}/{T} intervals", flush=True)

    df_results = pd.DataFrame(records).rename(columns={"lookahead_used": "future lookahead"})
    df_results["future lookahead"] = df_results["future lookahead"] - 1
    df_lol = pd.DataFrame(lol_logs)
    df_scd = pd.DataFrame(scd_events)
    df_line_viol = pd.DataFrame(all_line_violations)
    df_executed_commitment = pd.DataFrame(executed_commitment_records)

    return df_results, df_lol, df_scd, df_line_viol, df_executed_commitment

def summarize_results(df, df_lol, df_scd, df_line_viol, params):
    lol_period_count = int((df['LOL'] > LOL_TOL).sum()) if 'LOL' in df.columns else 0
    lol_total = float(df['LOL'].sum()) if 'LOL' in df.columns else 0.0
    lol_max = float(df['LOL'].max()) if 'LOL' in df.columns else 0.0
    emergency_trigger_hours = int((df['emergency_commit_triggered'] > 0).sum()) if 'emergency_commit_triggered' in df.columns else 0
    emergency_commit_total = int(df['emergency_commit_count'].sum()) if 'emergency_commit_count' in df.columns else 0
    pv_curt_total = float(df['curt_total'].sum()) if 'curt_total' in df.columns else 0.0
    pv_curt_max = float(df['curt_total'].max()) if 'curt_total' in df.columns else 0.0
    reserve_short_total = float(df['reserve_shortfall'].sum()) if 'reserve_shortfall' in df.columns else 0.0
    reserve_short_max = float(df['reserve_shortfall'].max()) if 'reserve_shortfall' in df.columns else 0.0
    solve_time_total_s = float(df['solve_time_seconds'].sum()) if 'solve_time_seconds' in df.columns else float('nan')
    solve_time_avg_s = float(df['solve_time_seconds'].mean()) if 'solve_time_seconds' in df.columns else float('nan')
    solve_time_max_s = float(df['solve_time_seconds'].max()) if 'solve_time_seconds' in df.columns else float('nan')
    line_violation_hours = int(df_line_viol['hour'].nunique()) if len(df_line_viol) and 'hour' in df_line_viol.columns else 0
    line_violation_total = float(df_line_viol['violation_MW'].sum()) if len(df_line_viol) and 'violation_MW' in df_line_viol.columns else 0.0
    line_violation_max = float(df_line_viol['violation_MW'].max()) if len(df_line_viol) and 'violation_MW' in df_line_viol.columns else 0.0
    line_slack_total = float(df['LineSlack_sum'].sum()) if 'LineSlack_sum' in df.columns else 0.0
    line_slack_max = float(df['LineSlack_max'].max()) if 'LineSlack_max' in df.columns else 0.0
    beta = params.beta
    return {
        'beta': beta,
        'LOL_period_count': lol_period_count,
        'LOL_total_MWh_equiv': lol_total,
        'LOL_max_MW': lol_max,
        'Emergency_trigger_hours': emergency_trigger_hours,
        'Emergency_commit_total': emergency_commit_total,
        'PV_curtailment_total_MWh_equiv': pv_curt_total,
        'PV_curtailment_max_MW': pv_curt_max,
        'Reserve_shortfall_total_MWh_equiv': reserve_short_total,
        'Reserve_shortfall_max_MW': reserve_short_max,
        'Total_solve_time_all_intervals_s': solve_time_total_s,
        'Total_solve_time_all_intervals_min': solve_time_total_s / 60.0,
        'Average_solve_time_per_interval_s': solve_time_avg_s,
        'Maximum_solve_time_per_interval_s': solve_time_max_s,
        'SCD_hours': int(df_scd['hour'].nunique()) if len(df_scd) and 'hour' in df_scd.columns else 0,
        'Line_violation_hours': line_violation_hours,
        'Line_violation_total_MW': line_violation_total,
        'Line_violation_max_MW': line_violation_max,
        'LineSlack_total_MW': line_slack_total,
        'LineSlack_max_MW': line_slack_max,
    }


def run_case(input_dir):
    case = load_case(input_dir)
    results = run_dispatch(case)
    df, df_lol, df_scd, df_line_viol, df_executed_commitment = results
    summary = pd.DataFrame([summarize_results(df, df_lol, df_scd, df_line_viol, case.params)])
    info = finalize_implementation_info(df, 1.0, float(case.params.future_lookahead + 1))
    output_dir = case.paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dispatch_results.xlsx": df,
        "lol_log.xlsx": df_lol,
        "scd_log.xlsx": df_scd,
        "line_violations.xlsx": df_line_viol,
        "executed_commitment.xlsx": df_executed_commitment,
        "run_summary.xlsx": summary,
        "implementation_information.xlsx": info,
    }
    for filename, table in tables.items():
        table.to_excel(output_dir / filename, index=False)
    print(f"Completed {len(df)} intervals. Results saved to {output_dir}")
    return results, summary, info
if __name__ == "__main__":
    input_dir = Path(__file__).resolve().parent / "Inputs"
    results, summary, implementation_info = run_case(input_dir)
