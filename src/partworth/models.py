"""Hierarchical multinomial logit (HB-MNL) for choice-based conjoint data.

Model, for respondent i, task t and alternative j::

    beta_i  = Delta' z_i + diag(tau) L eta_i,   eta_i ~ N(0, I)      (non-centred)
    L       ~ LKJCholesky(eta = lkj_eta),        tau ~ HalfNormal(tau_sd)
    U_itj   = x_itj' g(beta_i)
    y_it    ~ Categorical(softmax(U_it.))       over the alternatives shown

``g`` is the identity except for sign-constrained terms, where it is
``-exp(.)`` (negative) or ``exp(.)`` (positive): the lognormal used for price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .data import ChoiceData

if TYPE_CHECKING:
    from .results import HBResult

_PAD = -1e9  # utility for alternatives that were not shown


@dataclass(frozen=True)
class Priors:
    """Prior settings.

    Parameters
    ----------
    mu_sd
        Standard deviation of the normal prior on the population mean (intercept row
        of ``Delta``).
    delta_sd
        Standard deviation of the normal prior on covariate effects (other rows of
        ``Delta``). Covariates are centred, so the intercept stays the mean respondent.
    tau_sd
        Scale of the half-normal prior on the between-respondent standard deviations.
    lkj_eta
        Shape of the LKJ prior on the correlation matrix (1 = uniform; larger values
        shrink towards zero correlation).
    covariance
        ``"full"`` estimates correlations between coefficients; ``"diagonal"`` does not.
    """

    mu_sd: float = 2.0
    delta_sd: float = 1.0
    tau_sd: float = 1.0
    lkj_eta: float = 2.0
    covariance: Literal["full", "diagonal"] = "full"


class HierarchicalMNL:
    """Hierarchical Bayes multinomial logit.

    Parameters
    ----------
    data
        The estimation data. Only tasks in ``data.task_mask`` enter the likelihood.
    priors
        Prior settings.
    covariates
        Regress the population mean on the respondent covariates in ``data.Z``.
        When ``False`` only the intercept is used.
    """

    def __init__(self, data: ChoiceData, *, priors: Priors | None = None,
                 covariates: bool = True) -> None:
        self.data = data
        self.priors = priors or Priors()
        self.covariates = covariates
        self._model: pm.Model | None = None

    @property
    def coords(self) -> dict:
        d = self.data
        z_names = d.covariate_names if self.covariates else d.covariate_names[:1]
        return {
            "respondent": [str(r) for r in d.respondents],
            "coef": list(d.coef_names),
            "covariate": list(z_names),
        }

    def build(self) -> pm.Model:
        """Build (once) and return the PyMC model."""
        if self._model is not None:
            return self._model
        d, p = self.data, self.priors
        Z = d.Z if self.covariates else d.Z[:, :1]
        k, n_cov = d.n_params, Z.shape[1]
        ii, tt = np.nonzero(d.task_mask)
        X_obs = d.X[ii, tt]            # (M, J, K)
        mask_obs = d.mask[ii, tt]      # (M, J)
        y_obs = d.y[ii, tt]            # (M,)
        signs = d.codebook.sign_constraints()

        coords = self.coords | {"obs": np.arange(len(ii)), "coef_": list(d.coef_names)}
        with pm.Model(coords=coords) as model:
            Xd = pm.Data("X", X_obs)
            Zd = pm.Data("Z", Z, dims=("respondent", "covariate"))
            resp = pm.Data("respondent_index", ii, dims="obs")

            sd_row = np.r_[p.mu_sd, np.full(n_cov - 1, p.delta_sd)][:, None]
            delta = pm.Normal("Delta", 0.0, sigma=sd_row * np.ones((1, k)),
                              dims=("covariate", "coef"))
            if p.covariance == "full" and k > 1:
                chol, _, _ = pm.LKJCholeskyCov(
                    "chol_cov", n=k, eta=p.lkj_eta,
                    sd_dist=pm.HalfNormal.dist(p.tau_sd, shape=k), compute_corr=True,
                    store_in_trace=False,
                )
                cov = pm.Deterministic("cov", chol @ chol.T, dims=("coef", "coef_"))
                pm.Deterministic("tau", pt.sqrt(pt.diag(cov)), dims="coef")
                pm.Deterministic("corr", cov / pt.outer(pt.sqrt(pt.diag(cov)),
                                                        pt.sqrt(pt.diag(cov))),
                                 dims=("coef", "coef_"))
            else:
                tau = pm.HalfNormal("tau", p.tau_sd, dims="coef")
                chol = pt.diag(tau)

            eta = pm.Normal("eta", 0.0, 1.0, dims=("respondent", "coef"))
            raw = pm.Deterministic("beta_raw", Zd @ delta + eta @ chol.T,
                                   dims=("respondent", "coef"))
            beta = raw
            for col, sign in signs.items():
                val = -pt.exp(raw[:, col]) if sign == "negative" else pt.exp(raw[:, col])
                beta = pt.set_subtensor(beta[:, col], val)
            beta = pm.Deterministic("beta", beta, dims=("respondent", "coef"))

            util = (Xd * beta[resp][:, None, :]).sum(axis=-1)
            util = pt.switch(mask_obs, util, _PAD)
            pm.Categorical("y", logit_p=util, observed=y_obs, dims="obs")
        self._model = model
        return model

    def fit(
        self,
        *,
        draws: int = 1000,
        tune: int = 2000,
        chains: int = 4,
        sampler: Literal["nutpie", "pymc"] = "nutpie",
        target_accept: float = 0.9,
        random_seed: int | None = None,
        progressbar: bool = False,
        **kwargs,
    ) -> HBResult:
        """Draw from the posterior and return an :class:`~partworth.HBResult`.

        The default of 2000 tuning steps is deliberate: with little heterogeneity between
        respondents (small ``tau``), 500-1000 steps can leave chains disagreeing.
        """
        from .results import HBResult

        model = self.build()
        with model:
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, nuts_sampler=sampler,
                target_accept=target_accept, random_seed=random_seed,
                progressbar=progressbar, **kwargs,
            )
        result = HBResult(idata, self.data)
        result._warn_if_not_converged()
        return result

    def prior_predictive(self, samples: int = 500, random_seed: int | None = None):
        """Prior predictive draws (useful to check that priors give sensible shares)."""
        with self.build():
            return pm.sample_prior_predictive(draws=samples, random_seed=random_seed)
