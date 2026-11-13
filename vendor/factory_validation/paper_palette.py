"""Factory-figure colormaps aligned with the main-simulation paper palette.

Definitions mirror NewExperiment/scripts/viz_templates/nature_style.py so that
factory figures share the exact same color identity as Fig. 5, Fig. 8 and the
other main-simulation panels.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


PAPER_THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "paper_thermal",
    [
        "#24306F",
        "#2E6DA4",
        "#3FA6B3",
        "#88CFA7",
        "#F3E892",
        "#EFA45C",
        "#C94D45",
        "#7F1D2D",
    ],
)

PAPER_SIGNED_ERROR_CMAP = LinearSegmentedColormap.from_list(
    "paper_signed_error",
    [
        "#275B8E",
        "#8CB8D6",
        "#F7F7F7",
        "#E9A58E",
        "#B64342",
    ],
)

PAPER_RMSE_CMAP = LinearSegmentedColormap.from_list(
    "paper_rmse_low_to_high",
    [
        "#102F68",
        "#2F6FA8",
        "#78AFCF",
        "#C7DEEA",
        "#F5F3D4",
    ],
)


TEMP_CMAP = PAPER_THERMAL_CMAP
ABS_ERROR_CMAP = PAPER_RMSE_CMAP
DIFF_CMAP = PAPER_SIGNED_ERROR_CMAP
