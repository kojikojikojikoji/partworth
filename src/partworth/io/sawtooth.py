"""Sawtooth-style utility files.

The layout follows the public Lighthouse Studio documentation of exported utilities:
one row per respondent with a header row of labels, the respondent id, the fit
statistic (RLH) and one column per attribute level (raw utilities by default). Column
labels are ``"<attribute>: <level>"`` for categorical levels, ``"<attribute>"`` for
linear terms (utility per unit) and ``"None"`` for the none alternative. This is
"Sawtooth-style", defined from the public documentation; it has not been checked
against files produced by Sawtooth software.
"""

from __future__ import annotations

import os

import pandas as pd

from ..coding import NONE_COLUMN
from ..errors import SchemaError

RESPONDENT = "Respondent"
RLH = "RLH"


def utilities_frame(part_worths: pd.DataFrame, rlh: pd.Series | None = None) -> pd.DataFrame:
    """Flatten a part-worth table (columns ``(attribute, level)``) into Sawtooth layout."""
    labels = []
    for attribute, level in part_worths.columns:
        if attribute == NONE_COLUMN:
            labels.append("None")
        elif level == "slope":
            labels.append(str(attribute))
        else:
            labels.append(f"{attribute}: {level}")
    out = pd.DataFrame(part_worths.to_numpy(), columns=labels)
    if rlh is not None:
        out.insert(0, RLH, rlh.reindex(part_worths.index).to_numpy() * 1000)
    out.insert(0, RESPONDENT, part_worths.index.to_numpy())
    return out


def write_utilities(part_worths: pd.DataFrame, path: str | os.PathLike,
                    rlh: pd.Series | None = None) -> None:
    """Write utilities to CSV. RLH is written x1000, as in Sawtooth reports."""
    utilities_frame(part_worths, rlh).to_csv(path, index=False)


def read_utilities(path: str | os.PathLike) -> pd.DataFrame:
    """Read a Sawtooth-style utilities CSV into a ``(attribute, level)`` column table.

    The respondent id becomes the index. ``RLH`` (if present) is returned divided by
    1000 in the column ``("RLH", "RLH")``.
    """
    raw = pd.read_csv(path)
    if RESPONDENT not in raw.columns:
        raise SchemaError(f"no {RESPONDENT!r} column in {path}")
    raw = raw.set_index(RESPONDENT)
    cols = []
    for label in raw.columns:
        if label == RLH:
            cols.append((RLH, RLH))
        elif label == "None":
            cols.append((NONE_COLUMN, NONE_COLUMN))
        elif ": " in label:
            attribute, level = label.split(": ", 1)
            cols.append((attribute, level))
        else:
            cols.append((label, "slope"))
    out = raw.copy()
    out.columns = pd.MultiIndex.from_tuples(cols, names=["attribute", "level"])
    if (RLH, RLH) in out.columns:
        out[(RLH, RLH)] = out[(RLH, RLH)] / 1000
    return out
