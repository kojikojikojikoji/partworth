import numpy as np
import pandas as pd
import pytest

import partworth as pw


def build(df, attrs, **kw):
    return pw.ChoiceData.from_long(df, attributes=attrs, none="none", **kw)


def test_u01_shapes(small_long, small_attrs):
    d = build(small_long, small_attrs)
    # colour (effects, 2 cols) + price + NONE
    assert d.X.shape == (2, 2, 3, 4)
    assert d.y.shape == (2, 2)
    assert d.coef_names == ("colour[red]", "colour[green]", "price", "NONE")
    assert d.y.tolist() == [[0, 2], [1, 0]]


def test_u02_missing_column(small_long, small_attrs):
    with pytest.raises(pw.SchemaError, match="missing columns"):
        build(small_long.drop(columns="chosen"), small_attrs)


@pytest.mark.parametrize("value", [0, 1])
def test_u03_exactly_one_choice(small_long, small_attrs, value):
    df = small_long.copy()
    df.loc[(df.respondent == "a") & (df.task == 1), "chosen"] = value
    with pytest.raises(pw.ChoiceError):
        build(df, small_attrs)


def test_u04_ragged_mask(small_long, small_attrs):
    d = build(small_long, small_attrs)
    assert d.mask[0].sum() == 6
    assert d.mask[1].tolist() == [[True, True, False], [False, False, False]]
    assert d.n_tasks.tolist() == [2, 1]
    assert d.n_alternatives[1].tolist() == [2, 0]
    assert np.all(d.X[~d.mask] == 0)


def test_u05_none_rows(small_long, small_attrs):
    d = build(small_long, small_attrs)
    none_rows = d.X[0, 0, 2]
    assert none_rows.tolist() == [0, 0, 0, 1]
    assert d.X[d.mask][:, -1].sum() == 3  # three none rows in total


def test_u06_covariates(small_long, small_attrs):
    cov = pd.DataFrame({"age": [30, 50], "segment": ["x", "y"]}, index=["a", "b"])
    d = build(small_long, small_attrs, covariates=cov)
    assert d.covariate_names == ("Intercept", "age", "segment_y")
    np.testing.assert_allclose(d.Z[:, 1], [-10, 10])
    with pytest.raises(pw.SchemaError, match="missing for respondents"):
        build(small_long, small_attrs, covariates=cov.loc[["a"]])


def test_u07_subset_and_holdout(small_long, small_attrs):
    d = build(small_long, small_attrs)
    s = d.subset(["b"])
    assert s.n_respondents == 1 and s.respondents.tolist() == ["b"]
    assert s.tasks["i"].tolist() == [0]
    train, test = d.holdout([2])
    assert train.task_mask.tolist() == [[True, False], [True, False]]
    assert test.task_mask.tolist() == [[False, True], [False, False]]
    assert d.task_mask.sum() == 3  # original unchanged


def test_duplicated_keys(small_long, small_attrs):
    df = pd.concat([small_long, small_long.iloc[[0]]])
    with pytest.raises(pw.SchemaError, match="duplicated"):
        build(df, small_attrs)


def test_describe(small_long, small_attrs):
    out = build(small_long, small_attrs).describe()
    red = out[(out.attribute == "colour") & (out.level == "red")].iloc[0]
    assert (red.shown, red.chosen) == (2, 1)
    nn = out[out.attribute == "NONE"].iloc[0]
    assert (nn.shown, nn.chosen) == (3, 2)


@pytest.mark.network
def test_u08_camera():
    from partworth.datasets import load_camera
    d = load_camera()
    assert d.n_respondents == 332
    assert d.X.shape[1:3] == (16, 5)
    assert d.mask.all()
    assert d.coef_names[-1] == "NONE"
