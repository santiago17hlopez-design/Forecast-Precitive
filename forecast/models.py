"""Forecasting models.

Every model is a function with the same signature ``f(train, h) -> ndarray``:
given a monthly series, return the next ``h`` values. That uniform signature is
the whole point — adding a model means writing one function and registering it,
and nothing else in the codebase changes.

Models that need configuration (Fourier pairs, weighting half-life) read it from
the active run config rather than taking it as an argument, so the signature
stays fixed.

Models whose dependencies are not installed de-register themselves instead of
breaking the import. A missing optional model should cost you that model, not
the whole run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import active_config

M = 12  # seasonal period: monthly data

MODELS: dict[str, callable] = {}


def register(name):
    """Add a model to the registry under a display name."""

    def deco(f):
        MODELS[name] = f
        return f

    return deco


# --------------------------------------------------------------------- naive
# The baselines. Everything else has to beat these to justify its complexity,
# and on short or noisy series they frequently win.


@register("Naive")
def _naive(train, h):
    return np.repeat(float(train.iloc[-1]), h)


@register("Seasonal naive")
def _naive_seasonal(train, h):
    last_cycle = train.iloc[-M:].values
    return np.array([last_cycle[i % M] for i in range(h)], dtype=float)


@register("Seasonal naive + drift")
def _naive_drift(train, h):
    base = _naive_seasonal(train, h)
    if len(train) < 2 * M:
        return base
    drift = (float(train.iloc[-M:].mean()) - float(train.iloc[-2 * M:-M].mean())) / M
    return base + drift * np.arange(1, h + 1)


@register("Moving average 3m")
def _ma3(train, h):
    return np.repeat(float(train.iloc[-3:].mean()), h)


# ------------------------------------------------------------- exponential
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    def _holt(train, h, seasonal, damped=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                train,
                trend="add",
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=M,
                initialization_method="estimated",
            ).fit()
        return fit.forecast(h).values

    register("Holt-Winters additive")(lambda t, h: _holt(t, h, "add"))
    register("Holt-Winters multiplicative")(lambda t, h: _holt(t, h, "mul"))
    register("Holt-Winters damped")(lambda t, h: _holt(t, h, "add", damped=True))
except ImportError:  # pragma: no cover
    pass


# ------------------------------------------------------------------ SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def _sarima(train, h, order, sorder):
        """Fitted on logs, so the forecast cannot go negative.

        Production is a non-negative quantity. Fitting on the level lets the
        model hand back negative months, which is not a forecast, it is an
        artifact.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(
                np.log(train.clip(lower=1.0)),
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return np.exp(m.forecast(h).values)

    register("SARIMA(0,1,1)(0,1,1)")(lambda t, h: _sarima(t, h, (0, 1, 1), (0, 1, 1, M)))
    register("SARIMA(1,1,1)(0,1,1)")(lambda t, h: _sarima(t, h, (1, 1, 1), (0, 1, 1, M)))

    @register("SARIMA auto")
    def _sarima_auto(train, h):
        """Small grid search on AIC. Not a full auto-arima, deliberately.

        A handful of sensible orders covers what monthly business series
        actually need, and it stays fast enough to re-fit at every backtest
        origin — which is the constraint that matters here.
        """
        grid = [
            ((0, 1, 1), (0, 1, 1, M)),
            ((1, 1, 1), (0, 1, 1, M)),
            ((1, 1, 0), (1, 1, 0, M)),
            ((2, 1, 1), (0, 1, 1, M)),
        ]
        best, best_aic = None, np.inf
        y = np.log(train.clip(lower=1.0))
        for order, sorder in grid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=sorder,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best, best_aic = fit, fit.aic
            except Exception:
                continue
        if best is None:
            return _naive_seasonal(train, h)
        return np.exp(best.forecast(h).values)
except ImportError:  # pragma: no cover
    pass


# -------------------------------------------------------------- regression
def design_matrix(idx, kind="dummies", pairs=3):
    """Trend plus seasonality as a regression design matrix.

    ``dummies`` gives each month its own coefficient: flexible, but 11 extra
    parameters. ``fourier`` encodes the same seasonality in ``2 * pairs`` smooth
    terms, which is far steadier on short series — fewer parameters to estimate
    from the same amount of history.
    """
    t = np.arange(len(idx), dtype=float)
    if kind == "fourier":
        cols = [t]
        for k in range(1, pairs + 1):
            cols.append(np.sin(2 * np.pi * k * t / M))
            cols.append(np.cos(2 * np.pi * k * t / M))
        return np.column_stack(cols)
    months = pd.DatetimeIndex(idx).month
    dummies = np.zeros((len(idx), M - 1))
    for i, mo in enumerate(months):
        if mo > 1:
            dummies[i, mo - 2] = 1.0
    return np.column_stack([t, dummies])


def _future_index(train, h):
    return pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _ols_log(train, h, kind, weights=None):
    """Least squares on log(level), so growth is multiplicative."""
    y = np.log(np.clip(train.values, 1.0, None))
    X = design_matrix(train.index, kind, active_config().get("FOURIER_PAIRS", 3))
    X = np.column_stack([np.ones(len(X)), X])
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        w = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)

    full_idx = train.index.append(_future_index(train, h))
    Xf = design_matrix(full_idx, kind, active_config().get("FOURIER_PAIRS", 3))
    Xf = np.column_stack([np.ones(len(Xf)), Xf])[-h:]
    return np.exp(Xf @ beta)


register("Log regression + dummies")(lambda t, h: _ols_log(t, h, "dummies"))
register("Log regression + Fourier")(lambda t, h: _ols_log(t, h, "fourier"))


@register("Weighted regression")
def _weighted(train, h):
    """Exponentially down-weights old months.

    A business that changed two years ago should not be fitted as if those
    months still describe it. The half-life says how fast the past stops
    counting.
    """
    half_life = active_config().get("WEIGHT_HALF_LIFE", 24)
    age = np.arange(len(train))[::-1]
    w = 0.5 ** (age / half_life)
    return _ols_log(train, h, "fourier", weights=w)


# ----------------------------------------------------------------- ML-ish
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    def _supervised(train, h, estimator):
        y = np.log(np.clip(train.values, 1.0, None))
        X = design_matrix(train.index, "fourier", active_config().get("FOURIER_PAIRS", 3))
        estimator.fit(X, y)
        full_idx = train.index.append(_future_index(train, h))
        Xf = design_matrix(full_idx, "fourier", active_config().get("FOURIER_PAIRS", 3))[-h:]
        return np.exp(estimator.predict(Xf))

    register("Ridge")(lambda t, h: _supervised(t, h, Ridge(alpha=1.0)))
    register("Gradient Boosting")(
        lambda t, h: _supervised(
            t, h, GradientBoostingRegressor(random_state=active_config().get("SEED", 42))
        )
    )
except ImportError:  # pragma: no cover
    pass


@register("Ensemble")
def _ensemble(train, h):
    """Median of the stable members.

    Median rather than mean on purpose: when one member blows up — and on a
    series with a damaged stretch, one usually does — the median ignores it and
    the mean does not.
    """
    members = [
        "Seasonal naive",
        "Holt-Winters additive",
        "SARIMA(0,1,1)(0,1,1)",
        "Log regression + Fourier",
    ]
    preds = []
    for name in members:
        f = MODELS.get(name)
        if f is None:
            continue
        try:
            p = np.asarray(f(train, h), dtype=float)
            if np.all(np.isfinite(p)):
                preds.append(p)
        except Exception:
            continue
    if not preds:
        return _naive_seasonal(train, h)
    return np.median(np.vstack(preds), axis=0)


def available(requested):
    """Requested models that actually registered, in the requested order."""
    return [m for m in requested if m in MODELS]
