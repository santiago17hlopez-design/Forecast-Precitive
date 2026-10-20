"""Accuracy metrics, and the argument about which one decides.

The ranking is decided on RelMAE, not MASE. MASE divides by the seasonal naive's
error measured INSIDE the training window. When a series grows strongly, the
training window and the test window sit at different scales, and MASE inflates
for reasons that have nothing to do with the model. RelMAE compares both models
on the same out-of-sample points, so the scale cancels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mape(actual, pred):
    """Mean absolute percentage error, skipping near-zero actuals.

    MAPE divides by the actual, so a month near zero sends it to infinity. Those
    months say nothing about the model, so they are left out rather than allowed
    to decide the comparison.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    scale = np.median(np.abs(a))
    ok = np.abs(a) > max(scale * 0.10, 1e-9)
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100)


def wape(actual, pred):
    """Weighted absolute percentage error: total error over total volume.

    The honest headline number. Unlike MAPE it cannot be blown up by one small
    month, and it answers the question a business actually asks — across the
    whole period, how far off were we in money.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(a))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / denom * 100)


def bias(actual, pred):
    """Signed mean error. Tells you whether the model leans high or low."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(a))
    if denom <= 0:
        return float("nan")
    return float(np.sum(p - a) / denom * 100)


def summarize(bt: pd.DataFrame, baseline: str = "Seasonal naive") -> pd.DataFrame:
    """Per-model comparison table, sorted by RelMAE.

    RelMAE < 1 means the model beat the baseline on the same evaluation points.
    """
    if bt.empty:
        return pd.DataFrame()

    rows = []
    for name, g in bt.groupby("model"):
        rows.append(
            {
                "model": name,
                "n": len(g),
                "MAE": mae(g.actual, g.pred),
                "RMSE": rmse(g.actual, g.pred),
                "MAPE": mape(g.actual, g.pred),
                "WAPE": wape(g.actual, g.pred),
                "bias": bias(g.actual, g.pred),
            }
        )
    out = pd.DataFrame(rows).set_index("model")

    if baseline in out.index and out.loc[baseline, "MAE"] > 0:
        out["RelMAE"] = out["MAE"] / out.loc[baseline, "MAE"]
    else:
        out["RelMAE"] = np.nan

    return out.sort_values("RelMAE" if out["RelMAE"].notna().any() else "WAPE")


def diebold_mariano(bt: pd.DataFrame, model_a: str, model_b: str, h: int = 1):
    """Is the gap between two models real, or is it noise?

    Without this you rank by a third decimal place and call the winner. The test
    asks whether the difference in squared error is significantly different from
    zero across evaluation origins. |DM| > 1.96 is roughly the 5% line.

    Returns (dm_statistic, n_observations), or (nan, 0) when there is not
    enough overlap to compare.
    """
    a = bt[(bt.model == model_a) & (bt.h == h)].set_index("origin")
    b = bt[(bt.model == model_b) & (bt.h == h)].set_index("origin")
    common = a.index.intersection(b.index)
    if len(common) < 8:
        return float("nan"), len(common)

    ea = (a.loc[common, "actual"] - a.loc[common, "pred"]).values ** 2
    eb = (b.loc[common, "actual"] - b.loc[common, "pred"]).values ** 2
    d = ea - eb
    sd = np.std(d, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan"), len(common)
    return float(np.mean(d) / (sd / np.sqrt(len(d)))), len(common)
