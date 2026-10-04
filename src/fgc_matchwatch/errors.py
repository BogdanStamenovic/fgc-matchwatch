"""Errors shared across modules."""

from __future__ import annotations


class MatchwatchError(Exception):
    """Raised when fgc-matchwatch cannot complete an operation."""


class YouTubeBlocked(MatchwatchError):
    """YouTube refused us (bot check, sign-in, 403). A real stop: never worked around."""
