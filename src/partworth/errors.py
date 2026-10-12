"""Exceptions and warnings raised by partworth."""


class PartworthError(Exception):
    """Base class for all partworth errors."""


class SchemaError(PartworthError, ValueError):
    """The input table is missing columns, has wrong types, or duplicated keys."""


class ChoiceError(PartworthError, ValueError):
    """A choice task does not have exactly one chosen alternative."""


class CodingError(PartworthError, ValueError):
    """An attribute value is not one of the declared levels."""


class ConvergenceWarning(UserWarning):
    """MCMC diagnostics did not meet the thresholds (R-hat, ESS, divergences)."""
