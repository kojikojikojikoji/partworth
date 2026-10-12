"""partworth: hierarchical Bayes choice-based conjoint analysis in Python."""

from .coding import Categorical, Codebook, Linear
from .data import ChoiceData
from .errors import ChoiceError, CodingError, ConvergenceWarning, PartworthError, SchemaError

__version__ = "0.0.1.dev0"

__all__ = [
    "Categorical",
    "ChoiceData",
    "ChoiceError",
    "Codebook",
    "CodingError",
    "ConvergenceWarning",
    "Linear",
    "PartworthError",
    "SchemaError",
    "__version__",
]
