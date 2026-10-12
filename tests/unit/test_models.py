import numpy as np
import pytest

import partworth as pw
from partworth.datasets import simulate_cbc


@pytest.fixture(scope="module")
def sim():
    return simulate_cbc(30, 6, 3, {"brand": ["x", "y", "z"], "price": [1, 2, 3]},
                        price="price", seed=2)


def test_u15_model_shapes(sim):
    data, _ = sim
    m = pw.HierarchicalMNL(data).build()
    names = {v.name for v in m.free_RVs}
    assert {"Delta", "eta", "chol_cov"} <= names
    assert m.named_vars_to_dims["beta"] == ("respondent", "coef")
    assert len(m.coords["coef"]) == data.n_params


def test_u16_logp_finite(sim):
    data, _ = sim
    m = pw.HierarchicalMNL(data).build()
    assert np.isfinite(m.compile_logp()(m.initial_point()))


def test_u17_prior_predictive_price_negative(sim):
    data, _ = sim
    prior = pw.HierarchicalMNL(data).prior_predictive(samples=50, random_seed=0)
    price = prior["prior"]["beta"].sel(coef="price").to_numpy()
    assert (price < 0).all()


def test_u18_diagonal_has_no_correlation(sim):
    data, _ = sim
    m = pw.HierarchicalMNL(data, priors=pw.Priors(covariance="diagonal")).build()
    assert "corr" not in m.named_vars and "tau" in m.named_vars


def test_covariates_toggle(small_long, small_attrs):
    import pandas as pd
    cov = pd.DataFrame({"age": [30, 50]}, index=["a", "b"])
    data = pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none",
                                   covariates=cov)
    with_cov = pw.HierarchicalMNL(data).build()
    without = pw.HierarchicalMNL(data, covariates=False).build()
    assert len(with_cov.coords["covariate"]) == 2
    assert len(without.coords["covariate"]) == 1


def test_holdout_tasks_excluded_from_likelihood(sim):
    data, _ = sim
    train, _ = data.holdout([5])
    m = pw.HierarchicalMNL(train).build()
    assert len(m.coords["obs"]) == train.task_mask.sum() == 30 * 5


@pytest.mark.slow
def test_u19_seed_reproducible(sim):
    data, _ = sim
    kw = dict(draws=50, tune=50, chains=2, random_seed=7)
    a = pw.HierarchicalMNL(data).fit(**kw).beta_mean()
    b = pw.HierarchicalMNL(data).fit(**kw).beta_mean()
    np.testing.assert_allclose(a, b)
