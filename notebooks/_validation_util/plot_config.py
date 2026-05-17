"""
plot_config.py — Shared matplotlib configuration for the validation scripts.

Import this module at the top of any script that produces figures:

    import plot_config                       # rcParams applied on import
    from plot_config import new_fig, savefig_fig

Size conventions
----------------
Figures are sized to fit on *half the text width* of a standard single-column
arXiv preprint (≈ 6.5 in text width on US Letter with 1 in margins).

    HALF_WIDTH  = 3.25 in   → use for standalone / half-page figures
    FULL_WIDTH  = 6.50 in   → use for wide / two-panel figures

Font sizes are chosen so that labels read as ≈ 10 pt when the figure is
included at the target width in a LaTeX document.

Helpers
-------
new_fig(size="half", aspect=1.45)  → (fig, ax) with correct dimensions
savefig_fig(fig, stem, formats)    → save fig to stem.{fmt} for each fmt
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Size constants
# ---------------------------------------------------------------------------

HALF_WIDTH: float = 3.25   # inches — half of ~6.5 in arXiv text width
FULL_WIDTH: float = 6.50   # inches — full text width
WIDE_WIDTH: float = 7.00   # inches — full width + small bleed (2-panel)

# Default aspect ratios
_HALF_ASPECT: float = 1.45  # height = width / aspect  →  ~2.24 in
_FULL_ASPECT: float = 1.80  # height = width / aspect  →  ~3.61 in

# ---------------------------------------------------------------------------
# LaTeX / font settings (graceful fallback to mathtext)
# ---------------------------------------------------------------------------

_LATEX_PREAMBLE = r"\usepackage{amsmath}"
_USE_LATEX = False

try:
    subprocess.run(["latex", "--version"], capture_output=True, check=True, timeout=5)
    _USE_LATEX = True
except Exception:
    pass

_FONT_SETTINGS: dict = {
    "text.usetex":    _USE_LATEX,
    "font.family":    "serif" if _USE_LATEX else "sans-serif",
    "mathtext.fontset": "cm",          # used when usetex=False
}
if _USE_LATEX:
    _FONT_SETTINGS["text.latex.preamble"] = _LATEX_PREAMBLE

# ---------------------------------------------------------------------------
# Apply rcParams
# ---------------------------------------------------------------------------

# Retina output inside notebooks; silently ignored elsewhere.
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats("retina")
except Exception:
    pass

mpl.rcParams.update({
    # --- figure size & DPI ------------------------------------------------
    "figure.figsize":  (HALF_WIDTH, HALF_WIDTH / _HALF_ASPECT),
    "figure.dpi":       120,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",

    # --- font sizes (calibrated for HALF_WIDTH target) --------------------
    #   body text in arXiv ≈ 10-11 pt; figure text should be ≈ 8-9 pt
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "legend.fontsize":   7,
    "legend.title_fontsize": 7,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,

    # --- lines & markers --------------------------------------------------
    "lines.linewidth":      1.6,
    "lines.markersize":     4.5,
    "lines.markeredgewidth": 0.6,

    # --- axes -------------------------------------------------------------
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.7,

    # --- grid -------------------------------------------------------------
    "grid.alpha":       0.28,
    "grid.linewidth":   0.6,

    # --- legend -----------------------------------------------------------
    "legend.frameon":      True,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "0.85",
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    "legend.borderpad":     0.4,

    # --- layout -----------------------------------------------------------
    "figure.constrained_layout.use": True,

    **_FONT_SETTINGS,
})

# ---------------------------------------------------------------------------
# Colour / marker palettes
# ---------------------------------------------------------------------------
# Paul Tol "bright" — perceptually distinct, colorblind-safe.
# https://personal.sron.nl/~pault/#sec:qualitative
COLORS: list[str] = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey (fallback)
]

# Clean filled markers — readable at small sizes without edge clutter.
MARKERS: list[str] = ["o", "s", "^", "D", "P", "X"]

# Default scatter dot size (points²) for half-width figures.
SCATTER_SIZE: int = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_fig(
    size: str = "half",
    aspect: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kw,
) -> tuple:
    """Create a (fig, ax) pair sized for the target paper width.

    Parameters
    ----------
    size : "half" | "full" | "wide"
        Target width preset.
    aspect : float, optional
        width / height.  Defaults to 1.45 for "half", 1.80 for others.
    nrows, ncols : int
        Passed to ``plt.subplots``.
    **subplot_kw
        Extra keyword arguments forwarded to ``plt.subplots``.
    """
    widths = {"half": HALF_WIDTH, "full": FULL_WIDTH, "wide": WIDE_WIDTH}
    w = widths.get(size, HALF_WIDTH)
    if aspect is None:
        aspect = _HALF_ASPECT if size == "half" else _FULL_ASPECT
    h = w / aspect
    return plt.subplots(nrows, ncols, figsize=(w, h), **subplot_kw)


def savefig_fig(
    fig: mpl.figure.Figure,
    stem: Path | str,
    formats: Iterable[str] = ("png", "pdf"),
    *,
    dpi: int | None = None,
) -> None:
    """Save *fig* to ``stem.{fmt}`` for each format in *formats*.

    Parameters
    ----------
    fig : matplotlib Figure
    stem : path without extension
    formats : iterable of str, e.g. ``("png", "pdf")``
    dpi : int, optional — overrides savefig.dpi rcParam
    """
    stem = Path(stem)
    kw = dict(bbox_inches="tight", pad_inches=0)
    if dpi is not None:
        kw["dpi"] = dpi
    _pdfcrop = shutil.which("pdfcrop")
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, **kw)
        if fmt == "pdf" and _pdfcrop:
            subprocess.run([_pdfcrop, str(path), str(path)],
                           capture_output=True, check=False)
        print(f"  Saved: {path}")
