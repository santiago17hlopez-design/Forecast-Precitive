"""Rolling-origin backtesting.

The simulation is the real situation: stand at a month, know only the past,
forecast forward. The model is re-estimated from scratch at every origin —
carrying parameters across origins would be using information from the future.

Never random cross-validation. Shuffling months lets a model learn from later
months to predict earlier ones, which inflates every metric and tells you
nothing about next month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import MODELS


def run(series, model_names, min_window, h, window_type="expanding",
        rolling_length=60, log=None) -> pd.DataFrame:
    """One row per (model, origin, horizon) with the actual and the prediction."""
    n = len(series)
    origins = list(range(min_window, n - h + 1))
    rows = []

    if log:
        log.detail(f"  Evaluation origins : {len(origins)}")
        log.detail(f"  Horizons           : 1 to {h}")
        log.detail(f"  Window type        : {window_type}")
        log.detail(f"  Models compared    : {len(model_names)}")

    for i, cut in enumerate(origins):
        train = (
            series.iloc[:cut]
            if window_type == "expanding"
            else series.iloc[max(0, cut - rolling_length):cut]
        )
        for name in model_names:
            f = MODELS.get(name)
            if f is None:
                continue
            try:
                pred = np.asarray(f(train, h), dtype=float)
            except Exception:
                continue

            # Multiplicative Holt-Winters and seasonal drift return inf when the
            # series passes through near-zero months. That inf would contaminate
            # MAE, RMSE and WAPE for the model and end up written to the output
            # file. If an origin does not produce finite numbers, it does not count.
            if not np.all(np.isfinite(pred)):
                continue

            for step in range(1, h + 1):
                rows.append(
                    {
                        "model": name,
                        "origin": cut,
                        "h": step,
                        "actual": float(series.iloc[cut + step - 1]),
                        "pred": float(pred[step - 1]),
                    }
                )

        if log and ((i + 1) % 10 == 0 or i + 1 == len(origins)):
            log.detail(f"    origin {i + 1}/{len(origins)} done")

    return pd.DataFrame(rows)


def compare_windows(series, model_name, cfg, log=None):
    """Same model, expanding vs rolling window.

    Worth knowing. If the rolling window clearly wins, the series has a
    structural break and the old history is actively hurting the fit — which is
    a finding about the business, not about the model.
    """
    out = {}
    for window_type in ("expanding", "rolling"):
        bt = run(
            series,
            [model_name],
            cfg["MIN_WINDOW"],
            cfg["H_BACKTEST"],
            window_type=window_type,
            rolling_length=cfg["ROLLING_LENGTH"],
        )
        if bt.empty:
            continue
        from .metrics import wape

        out[window_type] = wape(bt.actual, bt.pred)

    if log and len(out) == 2:
        log.detail(
            f"  WAPE expanding {out['expanding']:.1f}%  ·  "
            f"rolling {out['rolling']:.1f}%"
        )
    return out


def interval_coverage(bt: pd.DataFrame, model_name: str, levels, log=None):
    """Did the intervals actually contain the truth as often as they claim?

    A nominal 80% band that covers 55% of observed months is not an 80% band.
    This is the check that stops a confidently wrong range from being published.
    """
    from .intervals import sigma_by_horizon
    from scipy.stats import norm

    g = bt[bt.model == model_name]
    if g.empty:
        return {}

    sigmas = sigma_by_horizon(bt, model_name, int(g.h.max()))
    out = {}
    for level in levels:
        z = norm.ppf(0.5 + level / 2)
        inside = 0
        for _, row in g.iterrows():
            s = sigmas.get(int(row.h))
            if s is None or not np.isfinite(s):
                continue
            lo = row.pred * (1 - z * s)
            hi = row.pred * (1 + z * s)
            if lo <= row.actual <= hi:
                inside += 1
        out[level] = inside / len(g) * 100

    if log:
        for level, pct in out.items():
            flag = "" if abs(pct - level * 100) <= 10 else "   <-- off"
            log.detail(f"  nominal {level:.0%}  observed {pct:5.1f}%{flag}")
    return out
