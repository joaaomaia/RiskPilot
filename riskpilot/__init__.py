from importlib.metadata import version

from .evaluation import BinaryPerformanceEvaluator, decile_analysis_plot
from .synthetic import LookAhead

__version__ = version("riskpilot")

__all__ = [
    "BinaryPerformanceEvaluator",
    "decile_analysis_plot",
    "LookAhead",
]
