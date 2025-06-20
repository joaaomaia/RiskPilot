from importlib.metadata import PackageNotFoundError, version

from .evaluation import BinaryPerformanceEvaluator, decile_analysis_plot
from .synthetic import LookAhead

try:
    __version__ = version("riskpilot")
except PackageNotFoundError:  # pragma: no cover - fallback during tests
    __version__ = "0.0.0"

__all__ = [
    "BinaryPerformanceEvaluator",
    "decile_analysis_plot",
    "LookAhead",
]
