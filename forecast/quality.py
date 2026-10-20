"""Data-damage detection, and what to do about it.

This module exists because of a specific failure mode that is easy to miss and
expensive to publish: a year that arrives empty from the source.

When a whole year is missing, the backtest stops measuring the forecast and
starts measuring how well each model reproduces the hole. The naive model wins,
the seasonal naive goes to ~100% error because it compares a real year against
an empty one, and the prediction bands come out in the hundreds of thousands of
millions. Every number downstream is then describing a missing file.

The warning prints regardless of verbosity, because a result like that must not
be published by someone who did not know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(ts, pattern="%Y-%m"):
    try:
        return pd.Timestamp(ts).strftime(pattern)
    except Exception:
        return str(ts)


def detect_anomalies(series, log) -> dict:
    """Flag near-empty stretches. Corrects nothing; reports only.

    Almost always this is the input, not the business.
    """
    s = series.dropna()
    if len(s) < 24:
        return {}

    level = s.rolling(12, min_periods=6).median().shift(1)

    # A month at zero or negative is broken data however you look at it, and it
    # lands before the rolling level has the 6 months it needs. Without the
    # second condition those months pass silently.
    collapsed = s[((level.notna()) & (level > 0) & (s < level * 0.25)) | (s <= 0)]

    yearly = s.groupby(s.index.year).sum()
    complete = yearly.iloc[:-1] if len(yearly) > 1 else yearly
    reference = float(complete.median()) if len(complete) else float("nan")
    thin = (
        yearly[yearly < reference * 0.25]
        if np.isfinite(reference) and reference > 0
        else yearly.iloc[0:0]
    )

    if not len(collapsed) and not len(thin):
        return {}

    # A year flagged as incomplete goes to the discard pile whole. Taking only
    # the months the rolling window manages to flag leaves the cut in the middle
    # of the broken year, and the remaining months — still a fraction of a normal
    # month — stay inside the training window. That is garbage in the training
    # set under a different name.
    broken_years = {int(y) for y in thin.index}
    months = set(collapsed.index)
    for date in s.index:
        if date.year in broken_years:
            months.add(date)
    months = sorted(months)

    log.milestone("")
    log.milestone("*" * 78)
    log.milestone("*** DATA ALERT: the series has near-empty stretches ***")
    log.milestone("*" * 78)
    for year, value in thin.items():
        log.milestone(
            f"    year {int(year)}: {value:>16,.0f}   "
            f"(median year of the series: {reference:,.0f})"
        )
    if months:
        log.milestone("    Discardable months (below 25% of level, or from a broken year):")
        log.milestone(
            "      " + ", ".join(_fmt(m) for m in months[:24])
            + (" ..." if len(months) > 24 else "")
        )
    log.milestone("")
    log.milestone("    Check the source file before the model: the usual cause is")
    log.milestone("    a whole annual file missing at the source.")
    log.milestone("    Until that is fixed, the model ranking, the WAPE and the")
    log.milestone("    confidence bands of this series are NOT publishable.")
    log.milestone("")

    return {
        "collapsed_months": [_fmt(m) for m in months],
        "incomplete_years": sorted(broken_years),
    }


def trim_damaged_span(series, warning: dict, cfg: dict, log):
    """Train only on the continuous stretch after the damage.

    An empty year in the middle of the history is not data, it is a file that
    never arrived. Inside the training window the model learns a 98% collapse
    and a 12,000% recovery that never happened, and from there everything —
    ranking, WAPE, bands — describes the hole rather than the business.

    Nothing is trimmed if what remains is too short to cover a cycle: in that
    case it is better to keep everything, with the alert in plain view.
    """
    if not cfg.get("TRIM_BROKEN_YEARS", True):
        return series

    months = (warning or {}).get("collapsed_months") or []
    if not months:
        return series

    last_bad = pd.Timestamp(max(months) + "-01")
    after = series[series.index > last_bad]
    discarded = series[series.index <= last_bad]

    log.milestone("")
    log.milestone("=" * 78)
    log.milestone("*** TRAINING WINDOW TRIMMED BECAUSE OF DAMAGED DATA ***")
    log.milestone("=" * 78)
    log.milestone(f"    Last damaged month : {_fmt(last_bad)}")

    years = (warning or {}).get("incomplete_years") or []
    if years:
        log.milestone("    Incomplete years   : " + ", ".join(str(y) for y in years))

    if len(after) < 12:
        log.milestone(
            f"    Only {len(after)} months would remain after the damage: not even"
        )
        log.milestone("    one cycle, so NOTHING is trimmed and training runs on")
        log.milestone("    everything, hole included. The result is not publishable.")
        log.milestone("=" * 78)
        cfg["_TRIM"] = None
        return series

    log.milestone(
        f"    Discarded stretch  : {_fmt(discarded.index.min())} to "
        f"{_fmt(discarded.index.max())} ({len(discarded)} months)"
    )
    log.milestone(
        f"    TRAINING FROM      : {_fmt(after.index.min())} ({len(after)} months)"
    )
    if len(after) < 24:
        log.milestone("")
        log.milestone("    NOTE: under 24 months there are not two full cycles, so")
        log.milestone("    whatever seasonality comes out is provisional.")
    log.milestone("")
    log.milestone("    Recovering the history means restoring the damaged year's")
    log.milestone("    file at the source; re-running the pipeline over the same")
    log.milestone("    empty file changes nothing.")
    log.milestone("=" * 78)
    log.milestone("")

    cfg["_TRIM"] = {
        "from": _fmt(after.index.min()),
        "training_months": int(len(after)),
        "discarded_months": int(len(discarded)),
        "last_damaged_month": _fmt(last_bad),
        "incomplete_years": years,
    }
    return after


def adjust_windows(series, cfg: dict, log) -> None:
    """Shrink the backtest window when the series came out short.

    With a 60-month minimum window, a series trimmed to 20 never reaches the
    backtest and the run aborts. The window is lowered just enough to produce
    evaluation origins, with a warning that the ranking is indicative. On a
    complete series this changes nothing.
    """
    n = len(series.dropna())
    needed = cfg["MIN_WINDOW"] + cfg["H_BACKTEST"] + 1
    if n >= needed:
        return

    h = max(3, min(cfg["H_BACKTEST"], n // 6))
    w = max(12, n - h - 5)
    origins = n - w - h + 1

    log.milestone("")
    log.milestone(f"[note] The series has {n} months; a standard backtest needs {needed}.")
    log.milestone(
        f"       Minimum window {cfg['MIN_WINDOW']} -> {w}   ·   "
        f"test horizon {cfg['H_BACKTEST']} -> {h}"
    )
    log.milestone(
        f"       {origins} evaluation origins remain: the model ranking is"
    )
    log.milestone("       indicative, not conclusive.")
    log.milestone("")

    cfg["MIN_WINDOW"] = w
    cfg["H_BACKTEST"] = h
    cfg["MIN_MONTHS_CATEGORY"] = min(cfg["MIN_MONTHS_CATEGORY"], max(6, n // 2))
    cfg["_WINDOWS_ADJUSTED"] = {"min_window": w, "h_backtest": h, "origins": int(origins)}
