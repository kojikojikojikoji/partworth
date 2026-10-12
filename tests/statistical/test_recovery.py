"""Parameter recovery on simulated data (S-01 to S-05). Run with --run-slow.

The main fixture uses the same data and settings as validation/recovery.py, whose
report is published in validation/results/recovery_report.md.
"""

import warnings

import numpy as np
import pytest

import partworth as pw
from partworth.datasets import simulate_cbc

pytestmark = pytest.mark.slow

MU = [0.6, -0.4, 0.4, 0.0, 0.3, -0.5]


@pytest.fixture(scope="module")
def fitted():
    data, truth = simulate_cbc(
        300, 12, 3, {"brand": ["x", "y", "z"], "size": ["s", "m", "l"], "price": [1, 2, 3, 4]},
        price="price", mu=MU, sd=0.6, corr=0.3, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pw.ConvergenceWarning)
        res = pw.HierarchicalMNL(data).fit(chains=4, random_seed=3)  # defaults, as recovery.py
    return data, truth, res


def _covered(arr, true, level=0.9):
    lo, hi = np.quantile(arr, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return (lo <= true) & (true <= hi)


def test_s01_population_mean_covered(fitted):
    _, truth, res = fitted
    delta = res.idata["posterior"]["Delta"].isel(covariate=0)
    draws = delta.stack(s=("chain", "draw")).transpose("s", ...).to_numpy()
    covered = _covered(draws, truth["mu"])
    assert covered.sum() >= len(MU) - 1, covered


def test_s02_sd_covered(fitted):
    _, truth, res = fitted
    tau = res.idata["posterior"]["tau"].stack(s=("chain", "draw")).transpose("s", ...)
    covered = _covered(tau.to_numpy(), np.sqrt(np.diag(truth["cov"])))
    assert covered.sum() >= len(MU) - 1, covered


def test_s04_individual_recovery(fitted):
    _, truth, res = fitted
    b = res.beta_mean()
    cors = [np.corrcoef(b[:, k], truth["beta"][:, k])[0, 1] for k in range(b.shape[1])]
    assert min(cors) > 0.5, cors


def test_s04b_individual_interval_coverage(fitted):
    _, truth, res = fitted
    draws = res.beta_draws()
    rate = _covered(draws, truth["beta"]).mean()
    assert 0.85 <= rate <= 0.95, rate


def test_s05_convergence(fitted):
    _, _, res = fitted
    d = res.diagnostics()
    assert d.ok, d.report()  # R-hat < 1.01, bulk ESS > 400, no divergences


def test_s05b_low_heterogeneity_converges():
    """Little heterogeneity (small tau) used to leave chains disagreeing with short tuning."""
    data, truth = simulate_cbc(
        60, 10, 3, {"brand": ["x", "y", "z"], "price": [1, 2, 3]},
        price="price", mu=[0.3, -0.2, 0.0, -0.5], sd=[0.06, 0.03, 0.2, 0.2], seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pw.ConvergenceWarning)
        res = pw.HierarchicalMNL(data, priors=pw.Priors(mu_sd=1.0, tau_sd=0.5)).fit(
            chains=4, random_seed=5)
    d = res.diagnostics()
    assert d.divergences == 0
    assert d.rhat_max < 1.02, d.report()
