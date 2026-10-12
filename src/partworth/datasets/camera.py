"""The ``camera`` data from the R package bayesm, fetched from CRAN at run time.

bayesm is licensed under the GPL, so neither its code nor its data are distributed with
partworth. :func:`load_camera_long` downloads the bayesm source package from CRAN into a
local cache the first time it is called, reads ``data/camera.rda`` with the ``rdata``
package, and returns a long table.

Source of the data: Allenby, Brazell, Howell and Rossi (2014), "Economic valuation of
product features", Quantitative Marketing and Economics 12(4).
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..coding import Categorical, Linear
from ..data import ChoiceData

BRANDS = ("canon", "sony", "nikon", "panasonic")
FEATURES = ("pixels", "zoom", "video", "swivel", "wifi")
_CRAN_META = "https://cloud.r-project.org/web/packages/bayesm/DESCRIPTION"
_CRAN_SRC = "https://cloud.r-project.org/src/contrib/bayesm_{version}.tar.gz"


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    base = os.environ.get("PARTWORTH_DATA") or os.environ.get("XDG_CACHE_HOME")
    return Path(base) / "partworth" if base else Path.home() / ".cache" / "partworth"


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "partworth (python)"})
    return urllib.request.urlopen(req, timeout=timeout)


def _fetch_rda(cache_dir: Path) -> Path:
    target = cache_dir / "bayesm_camera.rda"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _open(_CRAN_META, timeout=30) as r:
        desc = r.read().decode("utf-8")
    version = next(line.split(":", 1)[1].strip() for line in desc.splitlines()
                   if line.startswith("Version:"))
    with _open(_CRAN_SRC.format(version=version), timeout=120) as r:
        payload = r.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        member = tar.getmember("bayesm/data/camera.RData")
        data = tar.extractfile(member)
        assert data is not None
        target.write_bytes(data.read())
    return target


def load_camera_long(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return camera as a long table (respondent, task, alternative, chosen, none, ...).

    In bayesm each respondent has 16 tasks with 5 rows each; the fifth row is the
    outside ("none") option with all attributes zero. Brands are recoded to one
    ``brand`` column and the five binary features to ``"yes"``/``"no"``.
    """
    try:
        import rdata
    except ImportError as e:  # pragma: no cover
        raise ImportError("load_camera needs the 'rdata' package: "
                          "pip install 'partworth[datasets]'") from e
    path = _fetch_rda(_cache_dir(cache_dir))
    camera = rdata.read_rda(path)["camera"]
    frames = []
    for i, resp in enumerate(camera):
        X = np.asarray(resp["X"], dtype=float)
        y = np.asarray(resp["y"], dtype=int).ravel()
        n_alt = X.shape[0] // len(y)
        df = pd.DataFrame(X, columns=[*BRANDS, *FEATURES, "price"])
        df.insert(0, "respondent", i + 1)
        df.insert(1, "task", np.repeat(np.arange(1, len(y) + 1), n_alt))
        df.insert(2, "alternative", np.tile(np.arange(1, n_alt + 1), len(y)))
        df["chosen"] = (df["alternative"].to_numpy() == np.repeat(y, n_alt)).astype(int)
        brand = df[list(BRANDS)].to_numpy()
        df["none"] = brand.sum(axis=1) == 0
        df["brand"] = np.where(df["none"], None, np.array(BRANDS)[brand.argmax(axis=1)])
        for f in FEATURES:
            df[f] = np.where(df["none"], None, np.where(df[f] > 0, "yes", "no"))
        df["price"] = np.where(df["none"], np.nan, df["price"])
        frames.append(df.drop(columns=list(BRANDS)))
    return pd.concat(frames, ignore_index=True)


def load_camera(cache_dir: str | os.PathLike | None = None, *,
                price_sign: str | None = "negative") -> ChoiceData:
    """camera as :class:`~partworth.ChoiceData`.

    Coding follows bayesm so results can be compared: features are dummy coded against
    ``"no"``, brand is dummy coded against the outside option (through ``NONE``) with
    canon as reference, and price (hundreds of US dollars) enters linearly with a
    negative sign constraint and no centring or scaling. Pass ``price_sign=None`` for
    an unconstrained normal price coefficient, as in bayesm.
    """
    long = load_camera_long(cache_dir)
    attributes = {"brand": Categorical(BRANDS, coding="dummy", reference="canon")}
    attributes |= {f: Categorical(("no", "yes"), coding="dummy") for f in FEATURES}
    attributes["price"] = Linear(center=0.0, scale=1.0, sign=price_sign)
    return ChoiceData.from_long(long, attributes=attributes, none="none")
