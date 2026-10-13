import numpy as np
import pandas as pd
import pytest

import partworth as pw


@pytest.fixture
def data(small_long, small_attrs):
    return pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")


SCEN = pd.DataFrame({"product": ["P", "Q"], "colour": ["red", "blue"], "price": [20, 20]})


def test_u30_hand_computed_logit(data, make_result):
    # one respondent-draw: red=1, green=0, blue=-1, price 0 at centre, none=0
    beta = np.array([[[1.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]]])
    out = pw.MarketSimulator(make_result(data, beta)).shares(SCEN)
    e = np.exp([1.0, -1.0, 0.0])
    np.testing.assert_allclose(out["share"], e / e.sum())
    assert out["product"].tolist() == ["P", "Q", "NONE"]


def test_u31_shares_sum_to_one(data, make_result):
    rng = np.random.default_rng(1)
    res = make_result(data, rng.normal(size=(30, 2, 4)))
    out = pw.MarketSimulator(res).shares(SCEN)
    assert out["share"].sum() == pytest.approx(1.0)
    assert (out["lower"] <= out["share"]).all() and (out["share"] <= out["upper"]).all()


def test_u32_first_choice(data, make_result):
    beta = np.array([[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]])
    out = pw.MarketSimulator(make_result(data, beta), rule="first_choice").shares(
        SCEN, include_none=False)
    # respondent a prefers red (1 vs -1), b prefers blue (1 vs -1)
    np.testing.assert_allclose(out["share"], [0.5, 0.5])


def test_u33_iia_duplicate_splits_share(data, make_result):
    beta = np.array([[[0.5, 0.0, 0.0, 0.0]] * 2])
    sim = pw.MarketSimulator(make_result(data, beta))
    one = sim.shares(SCEN.iloc[[0]], include_none=True)
    two = sim.shares(pd.concat([SCEN.iloc[[0]], SCEN.iloc[[0]]]), include_none=True)
    # logit: the copy takes share from the original and from none in proportion
    e = np.exp(0.5)
    assert one["share"].iloc[0] == pytest.approx(e / (e + 1))
    assert two["share"].iloc[0] == pytest.approx(e / (2 * e + 1))


def test_u34_wtp_hand_computed(data, make_result):
    # price slope per unit = -2/10 = -0.2 ; red - blue = 1 - (-(1 + 0)) = 2  -> WTP 10
    beta = np.array([[[1.0, 0.0, -2.0, 0.0], [1.0, 0.0, -2.0, 0.0]]])
    out = pw.wtp(make_result(data, beta), attribute="colour", from_level="blue",
                 to_level="red")
    assert out["median"].iloc[0] == pytest.approx(10.0)


def test_price_curve_monotone(data, make_result):
    beta = np.array([[[0.0, 0.0, -2.0, 0.0], [0.0, 0.0, -1.0, 0.0]]])
    sim = pw.MarketSimulator(make_result(data, beta))
    curve = sim.price_curve(SCEN, product="P", grid=[10, 20, 30])
    assert curve["share"].is_monotonic_decreasing


def test_weights(data, make_result):
    beta = np.array([[[1.0, 0.0, 0.0, -9.0], [-1.0, 0.0, 0.0, -9.0]]])
    sim = pw.MarketSimulator(make_result(data, beta))
    only_a = sim.shares(SCEN, weights=pd.Series([1.0, 0.0], index=["a", "b"]))
    both = sim.shares(SCEN)
    assert only_a["share"].iloc[0] > both["share"].iloc[0]


def test_scenario_errors(data, make_result):
    sim = pw.MarketSimulator(make_result(data, np.zeros((2, 4))))
    with pytest.raises(pw.SchemaError):
        sim.shares(SCEN.drop(columns="price"))
    with pytest.raises(pw.CodingError):
        sim.shares(SCEN.assign(colour=["purple", "red"]))


def test_share_draws_hand_computed_rows(data, make_result):
    # two draws, both respondents alike: red=+1/-1, blue=-1/+1, price 0 at centre, none=0
    beta = np.array([[[1.0, 0.0, -1.0, 0.0]] * 2, [[-1.0, 0.0, 1.0, 0.0]] * 2])
    sim = pw.MarketSimulator(make_result(data, beta))
    draws = sim.share_draws(SCEN)
    e = np.exp([1.0, -1.0, 0.0])
    assert list(draws.columns) == ["P", "Q", "NONE"]
    np.testing.assert_allclose(draws.iloc[0], e / e.sum())
    np.testing.assert_allclose(draws.iloc[1], (e / e.sum())[[1, 0, 2]])
    without = sim.share_draws(SCEN, include_none=False)
    assert list(without.columns) == ["P", "Q"]
    np.testing.assert_allclose(without.iloc[0], e[:2] / e[:2].sum())


def test_share_draws_rows_align_across_scenarios(data, make_result):
    rng = np.random.default_rng(2)
    sim = pw.MarketSimulator(make_result(data, rng.normal(size=(40, 2, 4))))
    cheaper = SCEN.assign(price=[10, 20])
    base, new = sim.share_draws(SCEN), sim.share_draws(cheaper)
    pd.testing.assert_frame_equal(base, sim.share_draws(SCEN))
    out = sim.shares(SCEN)
    np.testing.assert_allclose(base.mean(axis=0), out["share"])
    # the price coefficient differs by draw, so the gain of P differs by draw too
    gain = new["P"] - base["P"]
    assert gain.nunique() > 1


def test_share_draws_weights(data, make_result):
    beta = np.array([[[1.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 1.0, 0.0]]])
    sim = pw.MarketSimulator(make_result(data, beta))
    first = pd.Series([1.0, 0.0], index=data.respondents)
    e = np.exp([1.0, -1.0, 0.0])
    np.testing.assert_allclose(sim.share_draws(SCEN, weights=first).iloc[0], e / e.sum())
