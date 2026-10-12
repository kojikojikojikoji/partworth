import numpy as np
import pandas as pd
import pytest

import partworth as pw
from partworth.io import attach_responses, wide_design_to_long

# shaped like ChoiceDesign's output: CS, alt{j}_{attr}, Block
DESIGN = pd.DataFrame({
    "CS": [1.0, 2.0, 3.0, 4.0],
    "alt1_colour": [0, 1, 2, 0], "alt1_price": [10, 20, 30, 20],
    "alt2_colour": [1, 2, 0, 2], "alt2_price": [30, 10, 20, 10],
    "Block": [1, 1, 2, 2],
})
LEVELS = {"colour": {0: "red", 1: "green", 2: "blue"}}


def test_u35_reshape():
    long = wide_design_to_long(DESIGN, none=True, levels=LEVELS)
    assert len(long) == 4 * 3
    first = long[long.task == 1]
    assert first["alternative"].tolist() == [1, 2, 3]
    assert first["colour"].tolist()[:2] == ["red", "green"]
    assert first["none"].tolist() == [False, False, True]
    assert pd.isna(first["colour"].iloc[2])
    assert long["task"].dtype.kind == "i"


def test_attach_and_build():
    long = wide_design_to_long(DESIGN, none=True, levels=LEVELS)
    answers = pd.DataFrame({"respondent": ["r1", "r1", "r2", "r2"], "task": [1, 2, 3, 4],
                            "block": [1, 1, 2, 2], "choice": [1, 3, 2, 1]})
    full = attach_responses(long, answers, block="block")
    assert full.groupby(["respondent", "task"])["chosen"].sum().eq(1).all()
    data = pw.ChoiceData.from_long(
        full, attributes={"colour": pw.Categorical(["red", "green", "blue"]),
                          "price": pw.Linear(sign="negative")}, none="none")
    assert data.X.shape == (2, 2, 3, 4)
    assert data.y.tolist() == [[0, 2], [1, 0]]


def test_attach_errors():
    long = wide_design_to_long(DESIGN)
    with pytest.raises(pw.ChoiceError):
        attach_responses(long, pd.DataFrame({"respondent": [1], "task": [1], "choice": [5]}))
    with pytest.raises(pw.SchemaError, match="not in the design"):
        attach_responses(long, pd.DataFrame({"respondent": [1], "task": [9], "choice": [1]}))
    with pytest.raises(pw.SchemaError, match="one row per respondent"):
        attach_responses(long, pd.DataFrame({"respondent": [1, 1], "task": [1, 1],
                                             "choice": [1, 2]}))


def test_pattern_errors():
    with pytest.raises(pw.SchemaError, match="no column"):
        wide_design_to_long(DESIGN, pattern="{attr}__{alt}")
    with pytest.raises(pw.SchemaError, match="lacks columns"):
        wide_design_to_long(DESIGN.drop(columns="alt2_price"))


def test_custom_pattern_without_cs():
    d = pd.DataFrame({"price_A": [1, 2], "price_B": [3, 4]})
    long = wide_design_to_long(d, pattern="{attr}_{alt}", block=None)
    assert long["task"].tolist() == [1, 1, 2, 2]
    assert np.array_equal(long["price"], [1, 3, 2, 4])
