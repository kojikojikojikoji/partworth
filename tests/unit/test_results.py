import numpy as np
import pandas as pd
import pytest

import partworth as pw
from partworth.io import read_utilities, write_utilities


@pytest.fixture
def data(small_long, small_attrs):
    return pw.ChoiceData.from_long(small_long, attributes=small_attrs, none="none")


BETA = np.array([[1.0, -0.5, -2.0, 0.3],    # colour[red], colour[green], price, NONE
                 [0.2, 0.4, -1.0, -0.1]])


def test_raw_part_worths(data, make_result):
    pw_ = make_result(data, BETA).part_worths()
    assert list(pw_.index) == ["a", "b"]
    assert pw_.loc["a", ("colour", "blue")] == pytest.approx(-0.5)
    # price coded as (x - 20) / 10, so per original unit the slope is beta / 10
    assert pw_.loc["a", ("price", "slope")] == pytest.approx(-0.2)


def test_u20_zero_centred_diffs(data, make_result):
    zc = make_result(data, BETA).part_worths("zc_diffs")
    np.testing.assert_allclose(zc["colour"].sum(axis=1), 0, atol=1e-9)
    span = 20.0  # price shown from 10 to 30
    colour_range = zc["colour"].max(axis=1) - zc["colour"].min(axis=1)
    price_range = zc[("price", "slope")].abs() * span
    np.testing.assert_allclose(colour_range + price_range, 200.0)


def test_u21_importances_sum_to_100(data, make_result):
    imp = make_result(data, BETA).importances()
    np.testing.assert_allclose(imp.sum(axis=1), 100.0)
    # respondent a: colour range 1.5, price range 0.2*20 = 4
    assert imp.loc["a", "colour"] == pytest.approx(100 * 1.5 / 5.5)


def test_u22_rlh_bounds_and_value(data, make_result):
    res = make_result(data, np.zeros((2, 4)))
    rlh = res.rlh()
    # all utilities zero: every alternative equally likely
    assert rlh["a"] == pytest.approx(1 / 3)
    assert rlh["b"] == pytest.approx(1 / 2)
    rlh2 = make_result(data, BETA).rlh()
    assert ((rlh2 > 0) & (rlh2 <= 1)).all()


def test_u16_predict_probabilities(data, make_result):
    pred = make_result(data, BETA).predict()
    assert len(pred) == 3
    assert pred["p_chosen"].between(0, 1).all()
    train, test = data.holdout([2])
    assert len(make_result(test, BETA).predict(test)) == 1


def test_u23_sawtooth_roundtrip(data, make_result, tmp_path):
    res = make_result(data, BETA)
    path = tmp_path / "utilities.csv"
    res.to_sawtooth_csv(path)
    header = path.read_text().splitlines()[0].split(",")
    assert header[:2] == ["Respondent", "RLH"]
    assert "colour: red" in header and "price" in header and header[-1] == "None"
    back = read_utilities(path)
    expected = res.part_worths()
    pd.testing.assert_frame_equal(back.drop(columns=("RLH", "RLH")), expected,
                                  check_names=False, check_index_type=False)
    np.testing.assert_allclose(back[("RLH", "RLH")], res.rlh().to_numpy())


def test_u24_write_without_rlh(data, make_result, tmp_path):
    path = tmp_path / "u.csv"
    write_utilities(make_result(data, BETA).part_worths(), path)
    assert path.read_text().splitlines()[0].startswith("Respondent,colour: red")


def test_u25_save_load(data, make_result, tmp_path):
    res = make_result(data, BETA)
    res.save(tmp_path / "fit.nc")
    back = pw.HBResult.load(tmp_path / "fit.nc", data)
    pd.testing.assert_frame_equal(back.part_worths(), res.part_worths())


def test_population_labels_log_scale(data, make_result):
    pop = make_result(data, BETA).population()
    assert pop.loc[("mean", "price"), "scale"] == "log(-coef per standardised unit)"
    assert pop.loc[("mean", "NONE"), "scale"] == "coef"


def test_diagnostics_report(data, make_result):
    rng = np.random.default_rng(0)
    res = make_result(data, rng.normal(size=(50, 2, 4)))
    d = res.diagnostics()
    assert isinstance(d.report(), str)
    assert d.divergences == 0
