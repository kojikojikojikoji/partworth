"""Regression tests for defects found in review on 2026-10-12."""

import numpy as np
import pandas as pd
import pytest

import partworth as pw
from partworth.io import read_utilities


@pytest.fixture
def data(small_long, small_attrs):
    return pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")


BETA = np.array([[1.0, -0.5, -2.0, 0.3], [0.2, 0.4, -1.0, -0.1]])


def test_predict_aligns_respondents(data, make_result):
    res = make_result(data, BETA)
    direct = res.predict().set_index(["respondent", "t"])["p_chosen"]
    reordered = res.predict(data.subset(["b", "a"])).set_index(["respondent", "t"])["p_chosen"]
    pd.testing.assert_series_equal(direct.sort_index(), reordered.sort_index())


def test_predict_rejects_unknown_respondents(data, make_result, small_long, small_attrs):
    other = small_long.assign(respondent=small_long["respondent"] + "x")
    d2 = pw.ChoiceData.from_long(other, attributes=small_attrs, none="none",
                                 codebook=data.codebook)
    with pytest.raises(pw.SchemaError, match="not in the estimation"):
        make_result(data, BETA).predict(d2)


def test_none_column_with_missing_values_rejected(small_long, small_attrs):
    df = small_long.assign(none=small_long["none"].where(small_long["none"], np.nan))
    with pytest.raises(pw.SchemaError, match="True/False"):
        pw.ChoiceData.from_long(df, attributes=small_attrs, none="none")


def test_integer_levels_with_none_rows(small_long):
    df = small_long.assign(size=[1, 2, np.nan, 2, 1, np.nan, 1, np.nan])
    d = pw.ChoiceData.from_long(df, attributes={"size": pw.Categorical([1, 2])},
                                none="none")
    assert d.coef_names == ("size[1]", "NONE")
    assert d.describe().set_index("level").loc["1", "shown"] == 3


def test_rlh_nan_without_estimation_tasks(data, make_result):
    with pytest.warns(UserWarning, match="no estimation task"):
        train, _ = data.holdout([1])
    rlh = make_result(train, BETA).rlh()
    assert np.isnan(rlh["b"]) and np.isfinite(rlh["a"])


def test_sawtooth_keeps_string_ids(small_long, small_attrs, make_result, tmp_path):
    df = small_long.assign(respondent=small_long["respondent"].map({"a": "001", "b": "002"}))
    d = pw.ChoiceData.from_long(df, attributes=small_attrs, none="none")
    res = make_result(d, BETA)
    res.to_sawtooth_csv(tmp_path / "u.csv")
    assert list(read_utilities(tmp_path / "u.csv").index) == ["001", "002"]


def test_price_curve_numpy_position(data, make_result):
    sim = pw.MarketSimulator(make_result(data, BETA))
    scen = pd.DataFrame({"colour": ["red", "blue"], "price": [20, 20]})
    out = sim.price_curve(scen, product=np.int64(0), grid=[10, 30])
    assert len(out) == 2


def test_max_draws_respected(data, make_result):
    rng = np.random.default_rng(0)
    sim = pw.MarketSimulator(make_result(data, rng.normal(size=(999, 2, 4))), max_draws=500)
    assert sim._beta.shape[0] == 500


def test_subset_duplicates(data):
    with pytest.raises(pw.SchemaError, match="unique"):
        data.subset(["a", "a"])


def test_full_covariance_matches_cholesky():
    """With Delta = 0, the empirical covariance of beta_raw over respondents ~ cov."""
    from partworth.datasets import simulate_cbc

    d, _ = simulate_cbc(4000, 1, 2, {"brand": ["x", "y", "z"]}, seed=0)
    m = pw.HierarchicalMNL(d).build()
    pt = m.initial_point()
    rng = np.random.default_rng(1)
    pt["eta"] = rng.normal(size=pt["eta"].shape)
    pt["chol_cov_cholesky-cov-packed__"] = rng.normal(scale=0.5, size=pt[
        "chol_cov_cholesky-cov-packed__"].shape)
    fn = m.compile_fn([m["beta_raw"], m["cov"]], inputs=m.value_vars, on_unused_input="ignore")
    raw, cov = fn(pt)
    np.testing.assert_allclose(np.cov(raw.T), cov, atol=0.15 * np.abs(cov).max())
