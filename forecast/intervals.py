"""Prediction intervals built from observed error, not from model theory.

The spread comes from the backtest, because that is the uncertainty that was
actually observed, including the part the model does not know about itself.

What is NOT used is the raw spread. See robust_sigma for why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import active_config


def robust_sigma(pred, actual) -> float:
    """Spread of relative error without two broken months dominating it.

    Relative error divides by the actual, so a month with near-zero volume
    sends it through the roof. A single stretch of damaged input can produce
    relative errors in the thousands of percent, and with an ordinary standard
    deviation those few months set the band for the entire series — the lower
    bound of the forecast ends up deeply negative. For a volume series that
    means nothing: it is not measuring uncertainty, it is measuring the hole in
    the data.

    Two defenses. Months whose actual is under 10% of the median are dropped —
    those measure the input, not the model — and the spread is computed with the
    scaled MAD (1.4826 x MAD, which equals sigma for normal errors) so a few
    extremes cannot drag it.

    Two readings are taken and the larger is kept, to bound without
    underestimating: the scaled MAD tolerates extremes but collapses toward zero
    when errors are nearly identical, while the 5-95 winsorized standard
    deviation keeps every point and only takes the edge off the tails. On a
    healthy series both land within a hair of the ordinary standard deviation.
    """
    cfg = active_config()
    floor = cfg.get("SIGMA_REL_FLOOR", 0.03)
    cap = cfg.get("SIGMA_REL_CAP", 0.60)

    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[finite], pred[finite]
    if len(actual) < 5:
        return float("nan")

    scale = float(np.median(np.abs(actual)))
    ok = np.abs(actual) > max(scale * 0.10, 1e-9)
    if ok.sum() < 5:
        return float("nan")

    e = (pred[ok] - actual[ok]) / actual[ok]

    mad = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    lo, hi = np.percentile(e, [5, 95])
    winsorized = float(np.std(np.clip(e, lo, hi), ddof=1))

    s = max(mad, winsorized)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(e, ddof=1))
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    return float(min(max(s, floor), cap))


def sigma_by_horizon(bt: pd.DataFrame, model_name: str, h_max: int) -> dict:
    """Observed relative error per horizon — the basis of the intervals.

    Horizons the backtest never reached are extrapolated with the square-root
    rule from the deepest horizon that was observed, and capped.
    """
    cfg = active_config()
    cap = cfg.get("SIGMA_REL_CAP", 0.60)

    g = bt[bt.model == model_name]
    sigmas = {}
    for h in sorted(g.h.unique()):
        gh = g[g.h == h]
        if len(gh) >= 5:
            s = robust_sigma(gh.pred.values, gh.actual.values)
            if np.isfinite(s):
                sigmas[int(h)] = s

    if not sigmas:
        return {h: 0.08 * np.sqrt(h) for h in range(1, h_max + 1)}

    deepest = max(sigmas)
    for h in range(1, h_max + 1):
        sigmas.setdefault(h, min(sigmas[deepest] * np.sqrt(h / deepest), cap))
    return sigmas


def _bound(center, z, sigma, lower: bool):
    """One interval bound. The lower one never goes below zero.

    Monthly volume is not negative. A negative lower bound only says the sigma
    went out of range, but on a report it reads as if the business could bill
    in reverse.
    """
    if not np.isfinite(center) or not np.isfinite(sigma):
        return np.nan
    v = center * (1 - z * sigma) if lower else center * (1 + z * sigma)
    if lower and center > 0:
        return max(v, 0.0)
    return v


def forecast_with_intervals(series, model_name, h, levels, bt=None) -> pd.DataFrame:
    """Central path of the chosen model, plus intervals at each level."""
    from .models import MODELS

    central = np.asarray(MODELS[model_name](series, h), dtype=float)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    out = pd.DataFrame({"central": central}, index=idx)
    out["h"] = range(1, h + 1)

    sigmas = sigma_by_horizon(bt, model_name, h) if bt is not None else {}
    if not sigmas:
        sigmas = {i: 0.08 * np.sqrt(i) for i in range(1, h + 1)}

    for level in levels:
        z = norm.ppf(0.5 + level / 2)
        tag = int(level * 100)
        out[f"lo_{tag}"] = [_bound(out.central.iloc[i], z, sigmas[i + 1], True) for i in range(h)]
        out[f"hi_{tag}"] = [_bound(out.central.iloc[i], z, sigmas[i + 1], False) for i in range(h)]

    out["sigma_rel"] = [sigmas[i + 1] for i in range(h)]
    return out


def aggregate_horizon(fc: pd.DataFrame, k: int, label: str, levels) -> dict:
    """Sum of the first k months, propagating uncertainty with sqrt(k).

    Monthly errors are partly independent — a high month offsets a low one — so
    the annual total is proportionally more reliable than any single month. The
    naive alternative, adding the monthly bounds, produces a yearly range so
    wide it is useless.
    """
    head = fc.head(k)
    total = float(head.central.sum())
    row = {"label": label, "months": k, "central": total}

    for level in levels:
        tag = int(level * 100)
        z = norm.ppf(0.5 + level / 2)
        sigma_agg = float(np.sqrt(np.mean(head.sigma_rel.values ** 2)) / np.sqrt(k))
        row[f"lo_{tag}"] = max(total * (1 - z * sigma_agg), 0.0)
        row[f"hi_{tag}"] = total * (1 + z * sigma_agg)
    return row
