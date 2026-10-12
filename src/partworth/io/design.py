"""Convert wide experimental designs (e.g. from ChoiceDesign) to partworth's long format.

ChoiceDesign (https://choicedesign.readthedocs.io/) returns one row per choice situation
with one column per alternative x attribute, such as ``alt1_price``, plus ``CS`` (choice
situation) and ``Block`` columns. :func:`wide_design_to_long` reshapes such a table and
:func:`attach_responses` joins respondents' answers to it, giving a table ready for
:meth:`partworth.ChoiceData.from_long`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..errors import ChoiceError, SchemaError


def wide_design_to_long(
    design: pd.DataFrame,
    *,
    pattern: str = "alt{alt}_{attr}",
    task: str = "CS",
    block: str | None = "Block",
    none: bool = False,
    levels: Mapping[str, Mapping] | None = None,
) -> pd.DataFrame:
    """Reshape a wide design to one row per task x alternative.

    Parameters
    ----------
    design
        One row per choice situation.
    pattern
        How attribute columns are named, with ``{alt}`` (alternative label) and
        ``{attr}`` (attribute name) placeholders.
    task, block
        Column names for the choice situation and (optional) block. If ``task`` is not a
        column, situations are numbered 1..n in row order.
    none
        Append a none alternative to every task.
    levels
        Optional attribute -> {code: label} mapping, e.g. ``{"colour": {0: "red"}}``.

    Returns
    -------
    DataFrame with columns ``task``, ``alternative``, (``block``), ``none`` and one
    column per attribute.
    """
    regex = re.compile(
        "^" + re.escape(pattern).replace(r"\{alt\}", "(?P<alt>.+?)")
        .replace(r"\{attr\}", "(?P<attr>.+)") + "$"
    )
    matched = {c: regex.match(str(c)) for c in design.columns if c not in (task, block)}
    matched = {c: m for c, m in matched.items() if m}
    if not matched:
        raise SchemaError(f"no column matches the pattern {pattern!r}")
    alts = list(dict.fromkeys(m["alt"] for m in matched.values()))
    attrs = list(dict.fromkeys(m["attr"] for m in matched.values()))
    lookup = {(m["alt"], m["attr"]): c for c, m in matched.items()}
    missing = [(a, b) for a in alts for b in attrs if (a, b) not in lookup]
    if missing:
        raise SchemaError(f"design lacks columns for (alternative, attribute): {missing[:5]}")

    task_ids = (design[task].to_numpy() if task in design.columns
                else np.arange(1, len(design) + 1))
    task_ids = [int(t) if float(t).is_integer() else t for t in task_ids]
    frames = []
    for j, alt in enumerate(alts, start=1):
        part = pd.DataFrame({"task": task_ids, "alternative": j})
        if block is not None and block in design.columns:
            part["block"] = design[block].to_numpy()
        part["none"] = False
        for attr in attrs:
            part[attr] = design[lookup[(alt, attr)]].to_numpy()
        frames.append(part)
    if none:
        part = frames[0][["task"] + (["block"] if "block" in frames[0] else [])].copy()
        part["alternative"] = len(alts) + 1
        part["none"] = True
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    for attr, mapping in (levels or {}).items():
        if attr not in long.columns:
            raise SchemaError(f"levels given for unknown attribute {attr!r}")
        mapped = long[attr].map(lambda v, m=mapping: m.get(v, m.get(_as_int(v), v)))
        long[attr] = mapped.where(~long["none"], np.nan)
    return long.sort_values(["task", "alternative"]).reset_index(drop=True)


def attach_responses(
    long_design: pd.DataFrame,
    responses: pd.DataFrame,
    *,
    respondent: str = "respondent",
    task: str = "task",
    choice: str = "choice",
    block: str | None = None,
) -> pd.DataFrame:
    """Join answers (one row per respondent x task, ``choice`` = alternative number).

    If ``block`` is given, each respondent's block is checked against the design.
    Returns the long table with ``respondent`` and a 0/1 ``chosen`` column.
    """
    need = [respondent, task, choice] + ([block] if block else [])
    miss = [c for c in need if c not in responses.columns]
    if miss:
        raise SchemaError(f"responses lack columns: {miss}")
    if responses.duplicated([respondent, task]).any():
        raise SchemaError("responses must have one row per respondent and task")
    r = responses[need].rename(columns={respondent: "respondent", task: "task",
                                        choice: "_choice"})
    keys = ["task"]
    if block:
        r = r.rename(columns={block: "block"})
        keys.append("block")
    merged = r.merge(long_design, on=keys, how="left", validate="one_to_many")
    if merged["alternative"].isna().any():
        bad = merged.loc[merged["alternative"].isna(), ["respondent", *keys]].head(3)
        raise SchemaError(f"answers refer to tasks not in the design, e.g.\n{bad}")
    merged["chosen"] = (merged["alternative"] == merged["_choice"]).astype(int)
    n = merged.groupby(["respondent", "task"])["chosen"].sum()
    if (n != 1).any():
        raise ChoiceError(f"{int((n != 1).sum())} answer(s) name an alternative that is "
                          "not in the task")
    cols = ["respondent", *[c for c in long_design.columns], "chosen"]
    return merged[cols].sort_values(["respondent", "task", "alternative"]).reset_index(
        drop=True)


def _as_int(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() else v
