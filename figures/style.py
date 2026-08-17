"""Shared plotting style for thesis figures.

Vector PDFs at the 14.5 cm text width, Arial to match the 11 pt body. Figure
scripts call apply_style, take their canvas from figure_size, and call
write_sidecar after saving so each PDF records the files it was built from.

Warm start is a dark blue from the Okabe and Ito set, cold start a mid grey.
They differ in lightness rather than hue, and solid against dashed repeats the
distinction, so the figure survives greyscale printing and deuteranopia.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import matplotlib as mpl


# Page geometry. The thesis text block is 14.5 cm wide.
TEXT_WIDTH_CM = 14.5
CM_PER_INCH = 2.54
TEXT_WIDTH_IN = TEXT_WIDTH_CM / CM_PER_INCH

# Point sizes as printed, since figures are placed at full text width with no scaling.
BASE_FONT_PT = 11.0
LABEL_FONT_PT = 8.0
TICK_FONT_PT = 7.0
LEGEND_FONT_PT = 8.0
PANEL_LETTER_PT = 8.0
ANNOTATION_PT = 7.0

# Separated by lightness rather than hue, and by line style, so the pair survives greyscale and deuteranopia.
COLOR_WARM = "#005A8F"
COLOR_COLD = "#999999"

# Faint versions used when a raw series is drawn behind a smoothed line.
RAW_ALPHA = 0.28
RAW_LINEWIDTH = 0.5

COLOR_REFERENCE = "#333333"

HAIRLINE = 0.4

WARM_STYLE = dict(color=COLOR_WARM, linestyle="-", linewidth=1.2)
COLD_STYLE = dict(color=COLOR_COLD, linestyle="--", linewidth=1.2)


def apply_style() -> None:
    """Set the global rcParams for every thesis figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": BASE_FONT_PT,
            "axes.labelsize": LABEL_FONT_PT,
            "axes.titlesize": LABEL_FONT_PT,
            "xtick.labelsize": TICK_FONT_PT,
            "ytick.labelsize": TICK_FONT_PT,
            "legend.fontsize": LEGEND_FONT_PT,
            # Vector PDF with embedded text rather than outlines, so it stays selectable in the thesis.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "figure.dpi": 200,
            # No frame on three sides, hairline elsewhere.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Gridlines never heavier than a hairline.
            "grid.color": "#CCCCCC",
            "grid.linewidth": HAIRLINE,
            "grid.alpha": 1.0,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.borderaxespad": 0.0,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
        }
    )


def figure_size(width_fraction: float = 1.0, aspect: float = 0.42):
    """Figure size in inches for a given fraction of the 14.5 cm text width.

    ``aspect`` is height divided by width.
    """
    width = TEXT_WIDTH_IN * width_fraction
    return (width, width * aspect)


def hairline_grid(ax, axis: str = "y") -> None:
    """Add a hairline grid behind the data on the requested axis."""
    ax.grid(True, axis=axis, linewidth=HAIRLINE, color="#CCCCCC")
    ax.set_axisbelow(True)


def sha256_of(path: str) -> str:
    """Hex sha256 of a file, read in chunks so large checkpoints are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Repository root, used only to relativise recorded paths, never to open anything.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_relative(path: str) -> str:
    """Path relative to the repository root, or unchanged if it lies outside.

    Uses a prefix test rather than ``os.path.relpath`` so a path outside the
    repository is left alone rather than rendered as a chain of ``..``.
    """
    absolute = os.path.abspath(path)
    root = _REPO_ROOT + os.sep
    if absolute.startswith(root):
        return absolute[len(root):]
    return path


def write_sidecar(
    pdf_path: str,
    sources,
    notes=None,
    extra=None,
) -> str:
    """Write the traceability sidecar beside a figure PDF.

    Records each source path with its size and sha256, plus any free text notes
    and an optional mapping of numbers a reader might check against the figure.

    Paths are recorded repository-relative, since the sidecar is tracked and an
    absolute one would name the machine that built it. Files are still opened by
    the absolute path passed in.

    Returns the sidecar path.
    """
    sidecar_path = os.path.splitext(pdf_path)[0] + ".sources.txt"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"Figure: {os.path.basename(pdf_path)}",
        f"Written: {stamp}",
        "",
        "Source files and sha256",
        "-----------------------",
    ]
    for source in sources:
        shown = _repo_relative(source)
        if os.path.exists(source):
            size = os.path.getsize(source)
            lines.append(f"{shown}")
            lines.append(f"  bytes  {size}")
            lines.append(f"  sha256 {sha256_of(source)}")
        else:
            lines.append(f"{shown}")
            lines.append("  MISSING at figure build time")
    if extra:
        lines += ["", "Values", "------"]
        for key, value in extra.items():
            lines.append(f"{key}: {value}")
    if notes:
        lines += ["", "Notes", "-----"]
        lines.extend(str(note) for note in notes)
    lines.append("")

    with open(sidecar_path, "w") as handle:
        handle.write("\n".join(lines))
    return sidecar_path
