from __future__ import annotations

from pathlib import Path

import plotly.graph_objs as go


def save_figure(fig: go.Figure, path: str | Path) -> str:
    """Save Plotly figure to the given path and return the path as str."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(p))
    return str(p)
