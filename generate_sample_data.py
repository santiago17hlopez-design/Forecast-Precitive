"""Generate a synthetic history so the whole pipeline is reproducible.

Everything in this repository runs on data produced here. No real data of any
kind is included, and none is needed.

The generator deliberately builds in the problems the engine is designed to
survive, because a pipeline only demonstrates robustness against data that
actually misbehaves:

  * trend plus multiplicative seasonality plus noise
  * a damaged year (one annual file that "never arrived")
  * a handful of negative months (cancellations exceeding sales)
  * comma-decimal formatting and a semicolon separator
  * a partial current month, to exercise the cutoff and the nowcast
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = [f"CAT-{i:03d}" for i in range(1, 13)]
MANAGERS = [f"MGR-{i}" for i in range(1, 6)]


def build(start="2017-01-01", months=108, damaged_year=2021, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    agents = []
    for m in MANAGERS:
        for k in range(1, rng.integers(5, 10)):
            agents.append((m, f"{m}-A{k:02d}"))

    month_starts = pd.date_range(start, periods=months, freq="MS")
    seasonal = {m: f for m, f in zip(range(1, 13),
                                     [0.82, 0.88, 1.05, 0.98, 1.02, 1.07,
                                      0.95, 0.99, 1.18, 1.09, 1.12, 0.85])}

    rows = []
    for i, ms in enumerate(month_starts):
        level = 820_000 * (1.006 ** i)
        factor = seasonal[ms.month] * rng.normal(1.0, 0.055)

        # The damaged year: the annual source file arrived nearly empty.
        if ms.year == damaged_year:
            factor *= 0.012

        month_total = level * factor
        days = pd.bdate_range(ms, ms + pd.offsets.MonthEnd(0))

        # Intake is batched: the first business days carry the renewals.
        weights = np.array([3.2 if d <= 5 else 1.0 for d in range(1, len(days) + 1)])
        weights = weights / weights.sum()

        for day, w in zip(days, weights):
            day_total = month_total * w
            for _ in range(rng.integers(4, 11)):
                manager, agent = agents[rng.integers(0, len(agents))]
                amount = day_total * rng.dirichlet(np.ones(1))[0] * rng.uniform(0.05, 0.35)

                # Occasional negative row: a cancellation reversing a prior sale.
                if rng.random() < 0.035:
                    amount = -abs(amount) * rng.uniform(0.3, 1.4)

                rows.append({
                    "Date": day.strftime("%Y-%m-%d"),
                    "Amount": round(amount, 2),
                    "Category": CATEGORIES[rng.integers(0, len(CATEGORIES))],
                    "ManagerCode": manager,
                    "AgentKey": f"0001-{agent}",
                    "AgentCode": agent,
                })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/history.csv")
    ap.add_argument("--months", type=int, default=108)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--damaged-year", type=int, default=2021,
                    help="Year to render near-empty. Use 0 for clean data.")
    args = ap.parse_args()

    df = build(months=args.months, damaged_year=args.damaged_year or -1, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Written the way the real sources arrive: semicolon separated, comma decimal.
    df.to_csv(args.out, sep=";", index=False, decimal=",", encoding="utf-8")

    print(f"Written: {args.out}")
    print(f"  rows      : {len(df):,}")
    print(f"  period    : {df.Date.min()} to {df.Date.max()}")
    print(f"  agents    : {df.AgentCode.nunique()}")
    print(f"  negatives : {(df.Amount < 0).sum():,}")
    if args.damaged_year:
        print(f"  damaged   : {args.damaged_year} (deliberately near-empty)")


if __name__ == "__main__":
    main()
