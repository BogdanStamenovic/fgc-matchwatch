"""fgc_matchwatch: find FIRST Global Challenge matches in the livestream and scout them."""

from __future__ import annotations

__version__ = "0.1.0"

from .cli import main
from .errors import MatchwatchError, YouTubeBlocked

__all__ = [
    "MatchwatchError",
    "YouTubeBlocked",
    "__version__",
    "main",
]
