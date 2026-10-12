"""Simulated choice-based conjoint data with known parameters (for recovery tests)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..coding import Categorical, Linear
from ..data import ChoiceData


def simulate_cbc(
    n_respondents: int,
    n_tasks: int,
    n_alternatives: int,
    attributes: Mapping[str, Sequence],
    *,
    none: bool = True,
    mu: Sequence[float] | None = None,
    sd: Sequence[float] | float = 0.8,
    corr: float = 0.0,
    price: str | None = None,
    seed: int = 0,
) -> tuple[ChoiceData, dict]:
    """Simulate responses from a hierarchical multinomial logit.

    Parameters
    ----------
    attributes
        Attribute name -> levels. String levels give an effects-coded
        :class:`~partworth.Categorical`; numeric levels give a :class:`~partworth.Linear`
        term centred on the mean of the levels and scaled by their standard deviation.
    mu, sd, corr
        Population mean, standard deviations and a common correlation for the coded
        coefficients. ``mu`` defaults to a draw from N(0, 1).
    price
        Name of a numeric attribute whose coefficient is negative for every respondent
        (lognormal, as in the model). Its ``mu``/``sd`` refer to the log of minus the
        coefficient.

    Returns
    -------
    data, truth
        ``truth`` has ``mu``, ``cov``, ``beta`` (N, K) in the coded space and the
        ``columns`` names.
    """
    rng = np.random.default_rng(seed)
    specs = {}
    for name, levels in attributes.items():
        if all(isinstance(v, int | float | np.number) for v in levels):
            vals = np.asarray(levels, dtype=float)
            sign = "negative" if name == price else None
            specs[name] = Linear(center=float(vals.mean()), scale=float(vals.std()) or 1.0,
                                 sign=sign)
        else:
            specs[name] = Categorical([str(v) for v in levels])

    rows = []
    for i in range(n_respondents):
        for t in range(n_tasks):
            for j in range(n_alternatives):
                row = {"respondent": i, "task": t, "alternative": j, "none": False}
                for name, levels in attributes.items():
                    row[name] = levels[rng.integers(len(levels))]
                rows.append(row)
            if none:
                rows.append({"respondent": i, "task": t, "alternative": n_alternatives,
                             "none": True, **{a: np.nan for a in attributes}})
    long = pd.DataFrame(rows)
    long["chosen"] = (long["alternative"] == 0).astype(int)  # placeholder

    data = ChoiceData.from_long(long, attributes=specs, none="none" if none else None)
    k = data.n_params
    mu_arr = rng.normal(0.0, 1.0, size=k) if mu is None else np.asarray(mu, dtype=float)
    sd_arr = np.broadcast_to(np.asarray(sd, dtype=float), (k,)).copy()
    corr_m = np.full((k, k), corr) + (1 - corr) * np.eye(k)
    cov = np.outer(sd_arr, sd_arr) * corr_m
    raw = rng.multivariate_normal(mu_arr, cov, size=n_respondents)
    beta = raw.copy()
    for col, sign in data.codebook.sign_constraints().items():
        beta[:, col] = (-1.0 if sign == "negative" else 1.0) * np.exp(raw[:, col])

    util = np.einsum("ntjk,nk->ntj", data.X, beta)
    util = np.where(data.mask, util, -np.inf)
    gumbel = rng.gumbel(size=util.shape)
    choice = np.argmax(util + gumbel, axis=2)

    pos = data.tasks.set_index(["respondent", "task"])
    alt_rank = long.groupby(["respondent", "task"])["alternative"].rank(method="dense") - 1
    i_idx = pos.loc[list(zip(long["respondent"], long["task"], strict=True)), "i"].to_numpy()
    t_idx = pos.loc[list(zip(long["respondent"], long["task"], strict=True)), "t"].to_numpy()
    long["chosen"] = (choice[i_idx, t_idx] == alt_rank.to_numpy().astype(int)).astype(int)

    data = ChoiceData.from_long(long, attributes=specs, none="none" if none else None)
    truth = {"mu": mu_arr, "cov": cov, "raw": raw, "beta": beta,
             "columns": data.coef_names}
    return data, truth
