"""The orchestrator: raw history in, one results file out.

    history.csv -> filters -> series -> models -> forecast_results.csv

One output file carries the historical series, the backtest, the forecast, the
allocation and each model's metrics. It used to be three files; one is better,
because three files meant three chances for a report to load a stale one and
disagree with itself.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import build_config, set_active
from . import backtest as bt_mod
from . import hierarchy, intervals, quality
from .log import Log
from .metrics import diebold_mariano, summarize
from .models import MODELS, available

# Column order of the output file is fixed here. A report that binds to column
# positions breaks silently when the order drifts, so it is defined once.
OUTPUT_COLUMNS = [
    "EntityCode", "kind", "level", "category", "date", "central",
    "lo_80", "hi_80", "lo_95", "hi_95", "sigma_rel",
    "ManagerCode", "AgentKey", "AgentCode", "share", "base_month",
    "model", "seasonal_index", "wape", "relmae", "bias",
]


def load_history(path, cfg, log) -> pd.DataFrame:
    """Read the source file, with the separator and decimal mark verified.

    The declared decimal mark is an expectation, not a fact. A file that says
    comma-decimal but arrives dot-decimal parses "1.234" as 1.234 instead of
    1234 and every figure is off by a factor of a thousand — silently, because
    nothing errors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    df = pd.read_csv(path, sep=cfg["SEP"], dtype=str, encoding=cfg["ENCODING"] or "utf-8")
    log.detail(f"  Rows read: {len(df):,}")

    col_value, col_date = cfg["COL_VALUE"], cfg["COL_DATE"]
    raw = df[col_value].astype(str).str.strip()

    def parse(series, decimal):
        s = series
        if decimal == ",":
            s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    comma_rows = raw.str.contains(r",\d+$", regex=True, na=False).sum()
    dot_rows = raw.str.contains(r"\.\d+$", regex=True, na=False).sum()
    detected = "," if comma_rows > dot_rows else "."
    if detected != cfg["DECIMAL"]:
        log.milestone(
            f"[note] Config says decimal '{cfg['DECIMAL']}' but the data looks like "
            f"'{detected}'. Using the data."
        )

    values = parse(raw, detected)

    # Last line of defence. If the guess was wrong, nearly everything fails to
    # parse and every downstream figure would be silently off by a factor of a
    # thousand — or absent entirely. Cheaper to try the other convention than to
    # publish that.
    if values.isna().mean() > 0.5:
        other = "." if detected == "," else ","
        retry = parse(raw, other)
        if retry.isna().mean() < values.isna().mean():
            log.milestone(
                f"[note] Decimal '{detected}' failed on "
                f"{values.isna().mean():.0%} of rows; falling back to '{other}'."
            )
            values, detected = retry, other

    df[col_value] = values

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    bad = df[col_date].isna().sum() + df[col_value].isna().sum()
    if bad:
        log.milestone(f"[note] {bad:,} rows dropped: unparseable date or value.")
    return df.dropna(subset=[col_date, col_value])


def build_series(df, cfg, log):
    """Monthly totals, with the current incomplete month optionally dropped.

    Leaving a half-finished month in the training data teaches the model that
    the business just collapsed.
    """
    col_date, col_value = cfg["COL_DATE"], cfg["COL_VALUE"]
    monthly = df.set_index(col_date)[col_value].resample("MS").sum()

    if cfg.get("START_DATE"):
        monthly = monthly[monthly.index >= pd.Timestamp(cfg["START_DATE"])]

    current_month_df = df.iloc[0:0]
    if cfg.get("AUTO_CUTOFF", True) and len(monthly):
        last = monthly.index[-1]
        last_day_in_data = df[col_date].max()
        month_end = last + pd.offsets.MonthEnd(0)
        if last_day_in_data < month_end:
            log.milestone(
                f"[cutoff] {last:%Y-%m} is incomplete (data to "
                f"{last_day_in_data:%Y-%m-%d}); excluded from training."
            )
            current_month_df = df[df[col_date].dt.to_period("M") == last.to_period("M")]
            monthly = monthly.iloc[:-1]

    return monthly, current_month_df


def validate_control_points(series, cfg, log):
    """The traffic light: compare against figures known from the report.

    Without these the run still works; you just lose the one check that catches
    a pipeline producing clean-looking numbers that disagree with what the
    business already published.
    """
    log.title("VALIDATION AGAINST CONTROL POINTS")
    monthly_cp = cfg.get("CONTROL_POINTS") or {}
    annual_cp = cfg.get("ANNUAL_CONTROL") or {}

    if not monthly_cp and not annual_cp:
        log.detail("  (No control points configured: validation skipped.)")
        log.detail("  Worth pulling two or three figures from the report before publishing.")
        return None

    tolerance = cfg.get("TOLERANCE_PCT", 1.0)
    all_ok = True

    for key, expected in monthly_cp.items():
        ts = pd.Timestamp(f"{key}-01")
        if ts not in series.index:
            log.detail(f"  {key}: not in the series — skipped")
            continue
        got = float(series.loc[ts])
        gap = abs(got - expected) / abs(expected) * 100 if expected else np.nan
        ok = np.isfinite(gap) and gap <= tolerance
        all_ok &= ok
        log.detail(f"  {key}: {got:>16,.1f} vs {expected:>16,.1f}  "
                   f"({gap:5.2f}%)  {'OK' if ok else '<-- OFF'}")

    for year, expected in annual_cp.items():
        got = float(series[series.index.year == int(year)].sum())
        gap = abs(got - expected) / abs(expected) * 100 if expected else np.nan
        ok = np.isfinite(gap) and gap <= tolerance
        all_ok &= ok
        log.detail(f"  {year}: {got:>16,.1f} vs {expected:>16,.1f}  "
                   f"({gap:5.2f}%)  {'OK' if ok else '<-- OFF'}")

    return all_ok


def seasonal_index(series):
    """Average month-of-year factor relative to the series level."""
    s = series.dropna()
    if len(s) < 24:
        return {}
    level = s.rolling(12, center=True, min_periods=6).mean()
    ratio = (s / level).dropna()
    return {int(m): float(v) for m, v in ratio.groupby(ratio.index.month).median().items()}


def run_forecast(input_path=None, output_path=None, entity_code="TOTAL",
                 overrides=None, verbosity="moderate", save=True) -> dict:
    """Full pipeline. Returns a summary dict; writes the results file."""
    cfg = build_config(overrides)
    set_active(cfg)
    log = Log(verbosity)

    input_path = input_path or cfg["INPUT_FILE"]
    output_path = output_path or cfg["OUTPUT_FILE"]

    log.title(f"FORECAST — {entity_code}", char="=")

    # ---- 1. load -------------------------------------------------------
    log.sub("1. Loading history")
    df = load_history(input_path, cfg, log)

    # ---- 2. series -----------------------------------------------------
    log.sub("2. Building the monthly series")
    series, current_df = build_series(df, cfg, log)
    log.detail(f"  Months: {len(series)}  ({series.index.min():%Y-%m} to "
               f"{series.index.max():%Y-%m})")

    # ---- 3. data quality ----------------------------------------------
    log.sub("3. Data quality")
    warning = quality.detect_anomalies(series, log)
    if warning:
        series = quality.trim_damaged_span(series, warning, cfg, log)
    quality.adjust_windows(series, cfg, log)

    if len(series) < cfg["MIN_WINDOW"] + cfg["H_BACKTEST"] + 1:
        log.alert("[stop] Not enough history to backtest. Nothing written.")
        return {"status": "insufficient_history", "months": int(len(series))}

    control_ok = validate_control_points(series, cfg, log)

    # ---- 4. backtest ---------------------------------------------------
    log.sub("4. Rolling-origin backtesting")
    active_models = available(cfg["ACTIVE_MODELS"])
    missing = set(cfg["ACTIVE_MODELS"]) - set(active_models)
    if missing:
        log.milestone(f"  [note] Not available in this environment: {', '.join(sorted(missing))}")

    bt = bt_mod.run(
        series, active_models, cfg["MIN_WINDOW"], cfg["H_BACKTEST"],
        window_type=cfg["WINDOW_TYPE"], rolling_length=cfg["ROLLING_LENGTH"], log=log,
    )
    if bt.empty:
        log.alert("[stop] The backtest produced no rows. Nothing written.")
        return {"status": "empty_backtest"}

    table = summarize(bt)
    log.detail("")
    log.detail(table.round(3).to_string())

    best = table.index[0]
    log.milestone(f"\n  Best model: {best}")

    # Is the gap over the runner-up real, or noise?
    if len(table) > 1:
        dm, n = diebold_mariano(bt, best, table.index[1])
        if np.isfinite(dm):
            verdict = "significant" if abs(dm) > 1.96 else "NOT significant"
            log.milestone(
                f"  vs {table.index[1]}: DM = {dm:+.2f} over {n} origins -> {verdict}"
            )
            if abs(dm) <= 1.96:
                log.milestone("  (The two are statistically tied. Prefer the simpler one.)")

    # ---- 5. intervals and coverage -------------------------------------
    log.sub("5. Forecast and intervals")
    fc = intervals.forecast_with_intervals(series, best, cfg["H_MAX"], cfg["LEVELS"], bt=bt)
    log.detail("")
    log.detail(fc.round(1).to_string())

    log.sub("Interval coverage (nominal vs observed)")
    coverage = bt_mod.interval_coverage(bt, best, cfg["LEVELS"], log=log)

    annual = intervals.aggregate_horizon(fc, min(12, cfg["H_MAX"]), "next 12 months",
                                         cfg["LEVELS"])
    log.milestone(f"\n  Next 12 months: {annual['central']:,.0f} "
                  f"(80%: {annual.get('lo_80', 0):,.0f} to {annual.get('hi_80', 0):,.0f})")

    # ---- 6. allocation --------------------------------------------------
    log.sub("6. Allocation by manager and agent")
    forecast_rows = [
        {"date": idx, "category": "TOTAL", "central": float(r.central), "kind": "Forecast"}
        for idx, r in fc.iterrows()
    ]
    allocated = hierarchy.allocate(forecast_rows, df, cfg["COL_CATEGORY"], cfg, log)
    hierarchy.check_reconciliation(allocated, forecast_rows, log=log)

    # ---- 7. assemble output ---------------------------------------------
    log.sub("7. Writing results")
    seasonal = seasonal_index(series)
    metrics_row = table.loc[best]

    rows = []
    for idx, value in series.items():
        rows.append({
            "EntityCode": entity_code, "kind": "History", "level": "Total",
            "category": "TOTAL", "date": idx, "central": float(value),
            "seasonal_index": seasonal.get(idx.month),
        })
    for idx, r in fc.iterrows():
        row = {
            "EntityCode": entity_code, "kind": "Forecast", "level": "Total",
            "category": "TOTAL", "date": idx, "central": float(r.central),
            "sigma_rel": float(r.sigma_rel), "model": best,
            "seasonal_index": seasonal.get(idx.month),
            "wape": float(metrics_row.WAPE), "relmae": float(metrics_row.RelMAE),
            "bias": float(metrics_row.bias),
        }
        for level in cfg["LEVELS"]:
            tag = int(level * 100)
            row[f"lo_{tag}"] = float(r[f"lo_{tag}"])
            row[f"hi_{tag}"] = float(r[f"hi_{tag}"])
        rows.append(row)
    for r in allocated:
        rows.append({"EntityCode": entity_code, **r})

    out = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[OUTPUT_COLUMNS]

    if save:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out.to_csv(output_path, sep=cfg["SEP"], index=False,
                   decimal=cfg["OUTPUT_DECIMAL"], encoding="utf-8-sig")
        log.milestone(f"  Written: {output_path}  ({len(out):,} rows)")

    return {
        "status": "ok",
        "best_model": best,
        "wape": float(metrics_row.WAPE),
        "relmae": float(metrics_row.RelMAE),
        "next_month": float(fc.central.iloc[0]),
        "next_12_months": annual["central"],
        "coverage": coverage,
        "control_points_ok": control_ok,
        "data_warning": warning or None,
        "trimmed": cfg.get("_TRIM"),
        "rows": len(out),
        "output_file": output_path,
    }
