"""Within-stratum duration associations; no causal or probability-model claims."""
from __future__ import annotations

import numpy as np
from scipy.stats import t


def _codes(values, n, name):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must be a one-dimensional array aligned with observations")
    lookup, codes = {}, []
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} contains a missing identity")
        try:
            if value != value:
                raise ValueError(f"{name} contains a missing identity")
            if value not in lookup:
                lookup[value] = len(lookup)
            codes.append(lookup[value])
        except TypeError as exc:
            raise ValueError(f"{name} identities must be hashable scalars") from exc
    return np.asarray(codes, dtype=np.int64), len(lookup)


def fixed_effect_linear(predictor, correct, strata, clusters, weights=None, covariates=None):
    """Estimate a focal within-stratum association with optional covariates.

    Fit ``y = stratum intercept + beta * predictor + covariates @ gamma`` with
    weighted demeaning. ``predictor`` may contain any finite real values;
    outcomes must be in [0, 1] and analytic weights must be positive and finite.
    ``covariates`` is None, an aligned one-dimensional vector, or an N-by-p
    matrix. No observations or covariate columns are silently removed.

    The returned slope is beta per unit of supplied predictor; ``slope_pp`` and
    its standard error/interval multiply beta by 100. CR1 uses N observations
    (not the weight sum), S stratum intercepts and P slope coefficients. The
    interval and two-sided p-value use t(G-1). A numerically rank-deficient
    within-stratum design returns ``not_identified``. These are observational
    associations, not causal effects or probability predictions.
    """
    predictor, correct = np.asarray(predictor, dtype=float), np.asarray(correct, dtype=float)
    if predictor.ndim != 1 or correct.ndim != 1 or len(predictor) != len(correct):
        raise ValueError("predictor and correct must be aligned one-dimensional arrays")
    n = len(predictor)
    if np.any(~np.isfinite(predictor)):
        raise ValueError("predictor must be finite")
    if np.any(~np.isfinite(correct)) or np.any((correct < 0) | (correct > 1)):
        raise ValueError("correct must be finite and within [0, 1]")
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != n or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be aligned, finite and strictly positive")
    if covariates is None:
        design = predictor[:, None]
    else:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        if covariates.ndim != 2 or covariates.shape[0] != n or np.any(~np.isfinite(covariates)):
            raise ValueError("covariates must be a finite aligned vector or N-by-p matrix")
        design = np.column_stack([predictor, covariates])
    p = design.shape[1]
    si, s = _codes(strata, n, "strata")
    ci, g = _codes(clusters, n, "clusters")
    residual_df = n - s - p
    result = {"status": "not_identified", "n": n, "nstrata": s, "nclusters": g,
              "effective_df": g - 1 if g >= 2 else None, "residual_df": residual_df,
              "slope": None, "slope_pp": None, "stderr_pp": None,
              "ci_lower_pp": None, "ci_upper_pp": None, "pvalue": None,
              "note": "Within-stratum predictor design is numerically rank deficient; "
                      "no association per unit of supplied predictor is identified."}
    if not n:
        result["note"] = "Empty cohort; no association per unit of supplied predictor can be estimated."
        return result
    sw = np.bincount(si, weights=w, minlength=s)
    xmean = np.column_stack([np.bincount(si, weights=w * design[:, j], minlength=s) / sw
                            for j in range(p)])
    ymean = np.bincount(si, weights=w * correct, minlength=s) / sw
    xr, yr = design - xmean[si], correct - ymean[si]
    squared_norm = np.sum(w[:, None] * xr ** 2, axis=0)
    original_scale = np.maximum(w.sum(), np.sum(w[:, None] * design ** 2, axis=0))
    if np.any(squared_norm <= np.finfo(float).eps * original_scale):
        return result
    # Scale columns before the rank check and solve, so different covariate
    # units do not themselves create an ill-conditioned numerical design.
    column_scale = np.sqrt(squared_norm)
    scaled = xr / column_scale
    weighted = scaled * np.sqrt(w)[:, None]
    singular = np.linalg.svd(weighted, compute_uv=False)
    if len(singular) < p or singular[-1] <= np.sqrt(np.finfo(float).eps) * singular[0]:
        return result
    bread = scaled.T @ (w[:, None] * scaled)
    coefficients = np.linalg.solve(bread, scaled.T @ (w * yr))
    slope = float(coefficients[0] / column_scale[0])
    result.update(status="variance_unavailable", slope=slope, slope_pp=100 * slope,
                  note="Association per unit of supplied predictor identified; "
                       "cluster uncertainty needs G >= 2 and N-S-P > 0.")
    if g < 2 or residual_df <= 0:
        return result
    residual = yr - scaled @ coefficients
    scores = np.column_stack([np.bincount(ci, weights=w * scaled[:, j] * residual, minlength=g)
                              for j in range(p)])
    inverse = np.linalg.inv(bread)
    correction = g / (g - 1) * (n - 1) / residual_df
    covariance = correction * inverse @ (scores.T @ scores) @ inverse
    variance = float(covariance[0, 0] / column_scale[0] ** 2)
    se = float(np.sqrt(max(0., variance)))
    critical = float(t.ppf(.975, g - 1))
    pvalue = float(2 * t.sf(abs(slope / se), g - 1)) if se else (1. if slope == 0 else 0.)
    result.update(status="ok", stderr_pp=100 * se,
                  ci_lower_pp=100 * (slope - critical * se),
                  ci_upper_pp=100 * (slope + critical * se), pvalue=pvalue,
                  note="Observed within-stratum association per unit of supplied predictor; analytic-weight WLS, "
                       "cluster CR1 and t(G-1) interval; not causal and not multiplicity-adjusted.")
    return result


def fixed_effect_slope(duration, correct, strata, clusters, weights=None):
    """Return the within-stratum association per doubling of positive duration.

    This is ``fixed_effect_linear(log2(duration), ...)`` without covariates.
    Durations must be finite and strictly positive; no observations are dropped.
    """
    duration = np.asarray(duration, dtype=float)
    if duration.ndim != 1:
        raise ValueError("duration must be a one-dimensional array")
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("duration must be finite and strictly positive")
    result = fixed_effect_linear(np.log2(duration), correct, strata, clusters, weights)
    result["note"] = result["note"].replace("per unit of supplied predictor", "per duration doubling")
    return result
