import os
import platform
import sys

# PyTensor <= 3.3.3 passes the removed "-ld64" flag to Apple's linker on recent macOS
# (fixed upstream in pymc-devs/pytensor#2432). Skip C compilation there; sampling uses
# nutpie/numba and is unaffected.
if sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0] or 0) >= 26:
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import partworth as pw  # noqa: E402


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


def fake_result(data: "pw.ChoiceData", beta: np.ndarray) -> "pw.HBResult":
    """An HBResult whose posterior draws are given directly; beta has shape (S, N, K)."""
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 2:
        beta = beta[None]
    s, n, k = beta.shape
    coords = {"chain": [0], "draw": np.arange(s), "respondent": [str(r) for r in data.respondents],
              "coef": list(data.coef_names), "covariate": ["Intercept"]}
    post = xr.Dataset(
        {"beta": (("chain", "draw", "respondent", "coef"), beta[None]),
         "Delta": (("chain", "draw", "covariate", "coef"), beta.mean(axis=1)[None, :, None, :]),
         "tau": (("chain", "draw", "coef"), beta.std(axis=1)[None])},
        coords=coords)
    stats = xr.Dataset({"diverging": (("chain", "draw"), np.zeros((1, s), dtype=bool))},
                       coords={"chain": [0], "draw": np.arange(s)})
    tree = xr.DataTree.from_dict({"posterior": post, "sample_stats": stats})
    return pw.HBResult(tree, data)


@pytest.fixture
def make_result():
    return fake_result
