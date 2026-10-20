"""Nowcasting the current month from a maturation curve.

Intake is batched, not even: the first business days concentrate renewals. So
"we are 40% through the month, therefore we are at 40% of the month" is wrong,
and wrong in a predictable direction.

The curve is the historical median share of the month's total accumulated by
each business day. Crucially it is validated backwards against months that have
already closed — without that step this is a hunch with a chart attached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def business_day_of_month(dates: pd.Series) -> pd.Series:
    """Which business day of its own month each date falls on."""
    out = []
    for d in pd.DatetimeIndex(dates):
        start = d.replace(day=1)
        out.append(int(np.busday_count(start.date(), d.date())) + 1)
    return pd.Series(out, index=dates.index)


def build_curve(df_closed: pd.DataFrame, col_date: str, col_value: str):
    """Median cumulative % of month total by business day."""
    d = df_closed.copy()
    d["month"] = d[col_date].dt.to_period("M")
    d = d.sort_values(col_date)
    d["bday"] = business_day_of_month(d[col_date])

    daily = d.groupby(["month", "bday"])[col_value].sum().unstack(fill_value=0)
    cumulative = daily.cumsum(axis=1)
    month_total = cumulative.iloc[:, -1]
    shares = cumulative.div(month_total, axis=0)
    return shares.median() * 100, cumulative, month_total


def validate_curve(curve, cumulative, month_total, cut_day=7, log=None):
    """Apply the curve to months that already closed and measure the error.

    This is what turns the nowcast from a guess into a number with a known
    error bar.
    """
    errors = []
    if cut_day not in cumulative.columns:
        return errors

    for month in cumulative.index[-13:-1]:
        partial = cumulative.loc[month, cut_day]
        frac = curve.get(cut_day, np.nan) / 100
        if not np.isfinite(frac) or frac <= 0:
            continue
        estimate = partial / frac
        actual = month_total[month]
        if actual <= 0:
            continue
        err = (estimate - actual) / actual * 100
        errors.append(abs(err))
        if log:
            log.detail(
                f"  {month}: partial {partial:>12,.0f} -> estimate {estimate:>14,.0f} | "
                f"actual {actual:>14,.0f} | error {err:+6.1f}%"
            )

    if errors and log:
        log.detail(f"\n  Mean absolute nowcast error at business day {cut_day}: "
                   f"{np.mean(errors):.1f}%")
    return errors


def nowcast(df_closed: pd.DataFrame, df_current: pd.DataFrame, cfg: dict,
            model_central=None, log=None) -> dict:
    """Estimate how the in-progress month will close."""
    col_date = cfg["COL_DATE"]
    col_value = cfg["COL_VALUE"]

    curve, cumulative, month_total = build_curve(df_closed, col_date, col_value)

    if log:
        log.title("MATURATION CURVE (historical median, cumulative % by business day)")
        for bday in [c for c in curve.index if c <= 12]:
            v = curve[bday]
            log.detail(f"  business day {int(bday):>2}: {v:6.1f}%   {'#' * int(v / 3)}")
        log.sub("VALIDATION: applying the curve to months already closed")

    errors = validate_curve(curve, cumulative, month_total, cut_day=7, log=log)

    if df_current.empty:
        if log:
            log.detail("\n  No data for the current month (every month is closed).")
        return {}

    dc = df_current.copy()
    dc["bday"] = business_day_of_month(dc[col_date])
    bday_now = int(dc["bday"].max())
    partial = float(dc[col_value].sum())
    frac = curve.get(bday_now, np.nan) / 100

    result = {
        "business_day": bday_now,
        "observed": partial,
        "expected_maturity_pct": float(frac * 100) if np.isfinite(frac) else None,
    }

    if log:
        log.title(f"NOWCAST OF THE CURRENT MONTH — {dc[col_date].max():%Y-%m}")
        log.detail(f"  Data through        : {dc[col_date].max():%Y-%m-%d} "
                   f"(business day {bday_now})")
        log.detail(f"  Observed to date    : {partial:,.0f}")

    if np.isfinite(frac) and frac > 0:
        estimate = partial / frac
        margin = np.mean(errors) / 100 if errors else 0.10
        result.update(
            {
                "estimated_close": float(estimate),
                "range_low": float(estimate * (1 - margin)),
                "range_high": float(estimate * (1 + margin)),
                "validation_error_pct": float(np.mean(errors)) if errors else None,
            }
        )
        if log:
            log.detail(f"  Expected maturity   : {frac * 100:.1f}% of the month")
            log.detail(f"  ESTIMATED CLOSE     : {estimate:,.0f}")
            log.detail(f"  Likely range        : {estimate * (1 - margin):,.0f}  to  "
                       f"{estimate * (1 + margin):,.0f}")
            if model_central is not None:
                log.detail(f"\n  Model forecast for that month: {model_central:,.0f}")

    return result
