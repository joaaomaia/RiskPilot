import importlib.util
import pytest


def has_shap() -> bool:
    return importlib.util.find_spec("shap") is not None


shap_available = pytest.mark.skipif(not has_shap(), reason="Optional dependency 'shap' not installed.")
