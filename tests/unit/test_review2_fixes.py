"""Regression tests for the code review of d47f6ff (2026-10-13)."""

import numpy as np
import pandas as pd
import pytest

import partworth as pw
from partworth.io import read_utilities, wide_design_to_long


def product_minus_none(levels: pd.DataFrame, colour: str, price: float | None) -> np.ndarray:
    u = levels[("colour", colour)].to_numpy().copy()
    if price is not None:
        u = u + levels[("price", "slope")].to_numpy() * price
    return u - levels[("NONE", "NONE")].to_numpy()


def test_zc_diffs_keep_product_vs_none_order_with_dummy_coding(small_long, make_result):
    attrs = {"colour": pw.Categorical(["red", "green", "blue"], coding="dummy")}
    data = pw.ChoiceData.from_long(small_long, attributes=attrs, none="none")
    # red = 0 (reference), green = 2, blue = 2, none = 1  ->  red is below none
    res = make_result(data, np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 1.0]]))
    raw = product_minus_none(res.part_worths(), "red", None)
    zc = product_minus_none(res.part_worths("zc_diffs"), "red", None)
    assert (raw < 0).all() and (zc < 0).all()
    assert np.allclose(np.sign(raw), np.sign(zc))


def test_exported_utilities_reproduce_product_vs_none(small_long, small_attrs, make_result,
                                                     tmp_path):
    data = pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")
    beta = np.array([[1.0, -0.5, -2.0, 0.3], [0.2, 0.4, -1.0, -0.1]])
    res = make_result(data, beta)
    # the model's own utility difference for a red product at price 25
    x = data.codebook.encode(pd.DataFrame({"colour": ["red"], "price": [25]}))[0]
    expected = beta @ x - beta[:, -1]
    res.to_sawtooth_csv(tmp_path / "u.csv")
    back = read_utilities(tmp_path / "u.csv")
    np.testing.assert_allclose(product_minus_none(back, "red", 25.0), expected)


def test_predict_rejects_different_linear_scaling(small_long, small_attrs, make_result):
    data = pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")
    other = pw.ChoiceData.from_long(
        small_long.assign(price=small_long["price"] * 10), none="none",
        attributes={"colour": small_attrs["colour"], "price": pw.Linear(sign="negative")})
    with pytest.raises(pw.SchemaError, match="coded differently"):
        make_result(data, np.zeros((2, 4))).predict(other)


def test_linear_importance_stable_under_subset(small_long, make_result):
    attrs = {"colour": pw.Categorical(["red", "green", "blue"]), "price": pw.Linear()}
    data = pw.ChoiceData.from_long(small_long, attributes=attrs, none="none")
    beta = np.array([[1.0, -0.5, -2.0, 0.3], [0.2, 0.4, -1.0, -0.1]])
    full = make_result(data, beta).importances()
    sub = make_result(data.subset(["b"]), beta[[1]]).importances()
    pd.testing.assert_series_equal(full.loc["b"], sub.loc["b"])


def test_design_alternative_numbers_follow_labels():
    design = pd.DataFrame({"CS": [1], "alt2_price": [30], "alt1_price": [10]})
    long = wide_design_to_long(design, block=None)
    assert long.set_index("alternative").loc[1, "price"] == 10
    assert long.set_index("alternative").loc[2, "price"] == 30


@pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0, -1.0]])
def test_simulator_rejects_bad_weights(small_long, small_attrs, make_result, weights):
    data = pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")
    sim = pw.MarketSimulator(make_result(data, np.zeros((2, 4))))
    scen = pd.DataFrame({"colour": ["red"], "price": [20]})
    with pytest.raises(pw.SchemaError, match="non-negative"):
        sim.shares(scen, weights=pd.Series(weights, index=["a", "b"]))


def test_likelihood_matches_hand_computation(small_long, small_attrs):
    data = pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")
    model = pw.HierarchicalMNL(data).build()
    point = model.initial_point()
    rng = np.random.default_rng(0)
    for name in point:
        point[name] = rng.normal(scale=0.5, size=np.shape(point[name]))
    beta_graph = model.replace_rvs_by_values([model["beta"]])[0]
    beta_fn = model.compile_fn(beta_graph, inputs=model.value_vars, on_unused_input="ignore")
    beta = beta_fn(point)
    ll_fn = model.compile_fn(model.logp(vars=[model["y"]], sum=True),
                             inputs=model.value_vars, on_unused_input="ignore")
    expected = 0.0
    for i, t in zip(*np.nonzero(data.task_mask), strict=True):
        alts = np.flatnonzero(data.mask[i, t])
        u = data.X[i, t, alts] @ beta[i]
        expected += u[list(alts).index(data.y[i, t])] - np.log(np.exp(u).sum())
    assert float(ll_fn(point)) == pytest.approx(expected, rel=1e-8)
