import numpy as np
import pandas as pd
import pytest

import partworth as pw
from partworth.coding import Codebook


def fit(attrs, df, has_none=False):
    return Codebook.fit(df, attrs, has_none)


DF = pd.DataFrame({"c": ["a", "b", "c", "a"], "p": [1.0, 2.0, 3.0, 2.0]})


def test_u09_effects():
    cb = fit({"c": pw.Categorical(["a", "b", "c"])}, DF)
    X = cb.encode(DF)
    assert cb.columns == ("c[a]", "c[b]")
    assert X.tolist() == [[1, 0], [0, 1], [-1, -1], [1, 0]]


def test_u10_dummy():
    cb = fit({"c": pw.Categorical(["a", "b", "c"], coding="dummy", reference="b")}, DF)
    assert cb.columns == ("c[a]", "c[c]")
    assert cb.encode(DF).tolist() == [[1, 0], [0, 0], [0, 1], [1, 0]]


def test_u11_effects_expand_sums_to_zero():
    cb = fit({"c": pw.Categorical(["a", "b", "c"])}, DF)
    out = cb.expand(np.array([[0.5, -0.2], [1.0, 2.0]]))
    np.testing.assert_allclose(out["c"].sum(axis=1), 0.0)
    assert out[("c", "c")].tolist() == pytest.approx([-0.3, -3.0])


def test_dummy_expand_reference_zero():
    cb = fit({"c": pw.Categorical(["a", "b", "c"], coding="dummy", reference="b")}, DF)
    out = cb.expand(np.array([0.4, -0.7]))
    assert out.iloc[0].tolist() == pytest.approx([0.4, 0.0, -0.7])


def test_u12_linear_center_scale():
    cb = fit({"p": pw.Linear()}, DF)
    z = cb.encode(DF)[:, 0]
    np.testing.assert_allclose(z.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(), 1)
    # a slope of 1 per standardised unit is 1/sd per original unit
    out = cb.expand(np.array([1.0]))
    assert out[("p", "slope")].iloc[0] == pytest.approx(1 / DF["p"].std(ddof=0))


def test_u13_unknown_level():
    cb = fit({"c": pw.Categorical(["a", "b", "c"])}, DF)
    with pytest.raises(pw.CodingError, match="'z'"):
        cb.encode(pd.DataFrame({"c": ["a", "z"]}))


def test_u14_roundtrip_level_labels():
    spec = pw.Categorical(["a", "b", "c"])
    cb = fit({"c": spec}, DF)
    X = cb.encode(DF)
    # recover the level from the coded row by matching against each level's code
    codes = cb.encode(pd.DataFrame({"c": list(spec.levels)}))
    back = [spec.levels[int(np.argmin(((codes - row) ** 2).sum(1)))] for row in X]
    assert back == DF["c"].tolist()


def test_sign_constraints_and_none():
    cb = fit({"c": pw.Categorical(["a", "b", "c"]), "p": pw.Linear(sign="negative")}, DF,
             has_none=True)
    assert cb.columns == ("c[a]", "c[b]", "p", "NONE")
    assert cb.sign_constraints() == {2: "negative"}


@pytest.mark.parametrize("bad", [dict(levels=["a"]), dict(levels=["a", "a"]),
                                 dict(levels=["a", "b"], reference="z"),
                                 dict(levels=["a", "b"], coding="other")])
def test_bad_specs(bad):
    with pytest.raises(pw.SchemaError):
        pw.Categorical(**bad)
