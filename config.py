"""Run configuration.

One dictionary holds every knob. Models read the active config through
``active_config()`` instead of taking parameters, which keeps the model
signature at ``(train, h)`` and makes adding a model a one-function change.
"""
from __future__ import annotations

import copy

CONFIG_BASE = {
    # --- Input ---
    "INPUT_FILE": "data/history.csv",
    "OUTPUT_FILE": "output/forecast_results.csv",
    "SEP": ";",
    "DECIMAL": ",",
    "ENCODING": None,  # None = detect

    # --- Columns in the input file ---
    "COL_DATE": "Date",
    "COL_VALUE": "Amount",
    "COL_CATEGORY": "Category",
    "COL_MANAGER": "ManagerCode",
    "COL_AGENT_KEY": "AgentKey",
    "COL_AGENT": "AgentCode",

    # --- Scope ---
    "START_DATE": None,          # None = from the first observation
    "AUTO_CUTOFF": True,         # drop the current month when it is incomplete

    # If a year arrives empty or broken at the source, train after that stretch
    # instead of swallowing it. See quality.trim_damaged_span.
    "TRIM_BROKEN_YEARS": True,

    # --- Horizons ---
    "H_MAX": 12,
    "H_BACKTEST": 6,

    # --- Backtesting ---
    "MIN_WINDOW": 60,
    "WINDOW_TYPE": "expanding",  # "expanding" | "rolling"
    "ROLLING_LENGTH": 60,

    # --- Models ---
    "ACTIVE_MODELS": [
        "Naive",
        "Seasonal naive",
        "Seasonal naive + drift",
        "Moving average 3m",
        "Holt-Winters additive",
        "Holt-Winters multiplicative",
        "Holt-Winters damped",
        "SARIMA(0,1,1)(0,1,1)",
        "SARIMA(1,1,1)(0,1,1)",
        "SARIMA auto",
        "Log regression + dummies",
        "Log regression + Fourier",
        "Weighted regression",
        "Ridge",
        "Gradient Boosting",
        "Ensemble",
    ],
    # Category-level series are shorter and noisier, so they get the subset of
    # models that stays stable on little history.
    "CATEGORY_MODELS": [
        "Seasonal naive",
        "Seasonal naive + drift",
        "Holt-Winters additive",
        "SARIMA(0,1,1)(0,1,1)",
        "Log regression + Fourier",
        "Log regression + dummies",
    ],
    "WEIGHT_HALF_LIFE": 24,
    "FOURIER_PAIRS": 3,

    # --- Allocation across the hierarchy ---
    # Each category forecast is split among managers and, inside each manager,
    # among their agents, using what each produced in that same category in the
    # SAME MONTH OF THE PREVIOUS YEAR. This is not a per-person forecast: it is
    # an allocation of one that already exists, so the report can be filtered by
    # manager or agent. See hierarchy.allocate.
    "ALLOCATE_HIERARCHY": True,
    "ALLOCATION_MONTHS_BACK": 12,

    # --- Categories ---
    "TOP_CATEGORIES": 8,
    "GROUP_TAIL": True,          # everything outside the top N becomes "Other"
    "MIN_MONTHS_CATEGORY": 48,

    # --- Intervals ---
    "LEVELS": [0.80, 0.95],
    "SIGMA_REL_FLOOR": 0.03,
    "SIGMA_REL_CAP": 0.60,

    # --- Validation ---
    # Known figures pulled from the published report. Without them the run still
    # works, you just lose the traffic light. See pipeline.validate_control_points.
    "CONTROL_POINTS": {},        # {"2025-09": 10590.4}
    "ANNUAL_CONTROL": {},        # {2025: 117124.4}
    "TOLERANCE_PCT": 1.0,

    # --- Output ---
    "OUTPUT_DECIMAL": ",",
    "SEED": 42,
}

_ACTIVE = dict(CONFIG_BASE)


def build_config(overrides: dict | None = None) -> dict:
    """A run config: the base with overrides applied, deep-copied."""
    cfg = copy.deepcopy(CONFIG_BASE)
    if overrides:
        cfg.update(overrides)
    return cfg


def set_active(cfg: dict) -> None:
    """Publish the config the models read from."""
    global _ACTIVE
    _ACTIVE = cfg


def active_config() -> dict:
    return _ACTIVE
