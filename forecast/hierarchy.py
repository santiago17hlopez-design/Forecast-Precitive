"""Allocating a forecast down an organizational hierarchy.

Each category's forecast is split among managers and, inside each manager, among
their agents, using the share each held in that same category in the SAME MONTH
OF THE PREVIOUS YEAR.

This is NOT a per-person forecast. Forecasting an individual agent's month from
their own history is not defensible — the series is short, noisy and full of
structural breaks (people join, leave, change portfolios). What this does is
allocate a forecast that already exists, so a report can be filtered by manager
or by agent without inventing precision that is not there.

Three decisions worth knowing about:

1. **Signs are preserved.** An agent who cancelled more than they sold enters
   with a negative share, and another can therefore exceed 100%. That is what
   makes the parts add back to exactly the total, instead of over-allocating
   among the positive ones.

2. **The agent's value is computed against the category total, not against
   their manager's.** Arithmetically identical — the manager total cancels —
   but it does not blow up when a manager closes a month at net ~zero. The
   share that gets *reported* is still the one within the manager, left empty
   when that net is zero.

3. **Allocated rows carry no prediction intervals.** The range of an allocated
   forecast is not a prediction interval, so those columns stay empty rather
   than being filled with a number that would be read as one.

Allocated rows are tagged ``kind="Allocation"`` deliberately. Report measures
that filter on ``kind="Forecast"`` will not pick them up, so no existing figure
changes; a measure that wants them has to ask for them explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _shares(df_base: pd.DataFrame, value_col: str, key_cols: list[str]) -> pd.DataFrame:
    """Net value and share per key, keeping the sign."""
    grouped = df_base.groupby(key_cols, dropna=False)[value_col].sum().reset_index()
    total = grouped[value_col].sum()
    grouped["share"] = grouped[value_col] / total if total != 0 else np.nan
    return grouped


def allocate(forecast_rows: list[dict], df_train: pd.DataFrame, category_col: str,
             cfg: dict, log) -> list[dict]:
    """Expand category-level forecast rows into manager and agent rows.

    ``forecast_rows`` are the central forecasts, each carrying at least
    ``date``, ``category`` and ``central``.
    """
    if not cfg.get("ALLOCATE_HIERARCHY", True) or not forecast_rows:
        return []

    col_manager = cfg["COL_MANAGER"]
    col_agent_key = cfg["COL_AGENT_KEY"]
    col_agent = cfg["COL_AGENT"]
    col_value = cfg["COL_VALUE"]
    col_date = cfg["COL_DATE"]
    months_back = int(cfg.get("ALLOCATION_MONTHS_BACK", 12))

    missing = [c for c in (col_manager, col_agent_key) if c not in df_train.columns]
    if missing:
        log.milestone(
            f"[note] The source does not carry {', '.join(missing)}: the forecast "
            "cannot be allocated by manager or agent."
        )
        return []

    out = []
    for row in forecast_rows:
        target_date = pd.Timestamp(row["date"])
        base_month = (target_date - pd.DateOffset(months=months_back)).to_period("M")

        in_base_month = df_train[col_date].dt.to_period("M") == base_month

        # "TOTAL" is the whole book, not a category literally named TOTAL. The
        # forecast can be produced at either level, and the base has to match
        # whichever one it was.
        if row["category"] == "TOTAL":
            base = df_train[in_base_month]
        else:
            base = df_train[in_base_month & (df_train[category_col] == row["category"])]

        if base.empty or base[col_value].sum() == 0:
            # No base month to allocate from. Saying nothing is better than
            # inventing a split; the caller logs how many rows fell back.
            continue

        total = base[col_value].sum()

        managers = _shares(base, col_value, [col_manager])
        for _, m in managers.iterrows():
            if not np.isfinite(m["share"]):
                continue
            out.append(
                {
                    **{k: v for k, v in row.items() if k != "central"},
                    "kind": "Allocation",
                    "level": "Manager",
                    col_manager: m[col_manager],
                    col_agent_key: None,
                    "central": float(row["central"] * m["share"]),
                    "share": float(m["share"]),
                    "base_month": str(base_month),
                }
            )

            agents_base = base[base[col_manager] == m[col_manager]]
            manager_net = agents_base[col_value].sum()
            agents = _shares(agents_base, col_value, [col_agent_key, col_agent])

            for _, a in agents.iterrows():
                # Against the category total, not the manager's: identical
                # arithmetic, but it survives a manager whose net is ~zero.
                share_of_total = a[col_value] / total if total != 0 else np.nan
                if not np.isfinite(share_of_total):
                    continue
                share_within = (
                    float(a[col_value] / manager_net) if manager_net != 0 else None
                )
                out.append(
                    {
                        **{k: v for k, v in row.items() if k != "central"},
                        "kind": "Allocation",
                        "level": "Agent",
                        col_manager: m[col_manager],
                        col_agent_key: a[col_agent_key],
                        col_agent: a[col_agent],
                        "central": float(row["central"] * share_of_total),
                        "share": share_within,
                        "base_month": str(base_month),
                    }
                )

    if log:
        managers_n = sum(1 for r in out if r["level"] == "Manager")
        agents_n = sum(1 for r in out if r["level"] == "Agent")
        log.detail(f"  Allocated rows: {managers_n} manager · {agents_n} agent")

    return out


def check_reconciliation(allocated: list[dict], forecast_rows: list[dict],
                         tolerance=0.01, log=None) -> bool:
    """The allocation must add back to the forecast it came from.

    This is the check that makes the whole thing trustworthy. If agent rows do
    not sum to their category's forecast, the split is wrong and the report will
    quietly disagree with itself depending on which filter is applied.
    """
    if not allocated:
        return True

    agents = pd.DataFrame([r for r in allocated if r["level"] == "Agent"])
    if agents.empty:
        return True

    source = pd.DataFrame(forecast_rows)
    summed = agents.groupby(["date", "category"])["central"].sum()
    expected = source.set_index(["date", "category"])["central"]

    common = summed.index.intersection(expected.index)
    if len(common) == 0:
        return True

    diff = (summed.loc[common] - expected.loc[common]).abs()
    scale = expected.loc[common].abs().clip(lower=1.0)
    worst = float((diff / scale).max())

    ok = worst <= tolerance
    if log:
        if ok:
            log.detail(f"  Allocation reconciles (worst gap {worst:.4%})")
        else:
            log.milestone(
                f"  *** ALLOCATION DOES NOT RECONCILE: worst gap {worst:.2%} ***"
            )
    return ok
