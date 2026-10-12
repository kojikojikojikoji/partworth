import pandas as pd
import pytest

import partworth as pw


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="run tests that download data")
    parser.addoption("--run-slow", action="store_true", help="run MCMC tests")


def pytest_collection_modifyitems(config, items):
    for name in ("network", "slow"):
        if config.getoption(f"--run-{name}"):
            continue
        skip = pytest.mark.skip(reason=f"needs --run-{name}")
        for item in items:
            if name in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def small_long() -> pd.DataFrame:
    """Two respondents; respondent 'b' has one task fewer and a task with two options."""
    rows = [
        # respondent, task, alternative, colour, price, none, chosen
        ("a", 1, 1, "red", 10, False, 1),
        ("a", 1, 2, "blue", 20, False, 0),
        ("a", 1, 3, None, None, True, 0),
        ("a", 2, 1, "green", 30, False, 0),
        ("a", 2, 2, "red", 20, False, 0),
        ("a", 2, 3, None, None, True, 1),
        ("b", 1, 1, "blue", 10, False, 0),
        ("b", 1, 2, None, None, True, 1),
    ]
    return pd.DataFrame(rows, columns=["respondent", "task", "alternative", "colour", "price",
                                       "none", "chosen"])


@pytest.fixture
def small_attrs():
    return {"colour": pw.Categorical(["red", "green", "blue"]),
            "price": pw.Linear(center=20, scale=10, sign="negative")}
