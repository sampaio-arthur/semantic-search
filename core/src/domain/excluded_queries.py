"""Centralized list of queries excluded from evaluation.

Previously contained 11 queries excluded from the BEIR trec-covid dataset.
The exclusion mechanism has been deactivated — the system now uses all
queries from the dataset without filtering.
"""

from __future__ import annotations

EXCLUDED_QUERY_TEXTS: frozenset[str] = frozenset()


def is_excluded_query(query_text: str) -> bool:
    """Always returns False — query exclusion is deactivated."""
    return False
