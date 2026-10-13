"""partworth: hierarchical Bayes choice-based conjoint analysis in Python."""

from .coding import Categorical, Codebook, Linear
from .data import ChoiceData
from .errors import ChoiceError, CodingError, ConvergenceWarning, PartworthError, SchemaError
from .models import HierarchicalMNL, Priors
from .results import Diagnostics, HBResult
from .simulate import MarketSimulator, wtp

__version__ = "0.0.1"

__all__ = [
    "Categorical",
    "ChoiceData",
    "ChoiceError",
    "Codebook",
    "Diagnostics",
    "HBResult",
    "HierarchicalMNL",
    "Priors",
    "CodingError",
    "ConvergenceWarning",
    "Linear",
    "MarketSimulator",
    "PartworthError",
    "SchemaError",
    "__version__",
    "wtp",
]
