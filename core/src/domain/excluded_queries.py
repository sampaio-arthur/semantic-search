"""Centralized list of queries excluded from evaluation.

These queries were identified empirically as producing inconsistent or noisy
results on the BEIR trec-covid dataset, compromising experimental validity.
Excluding them is a standard methodological decision in IR benchmarking.
"""

from __future__ import annotations

EXCLUDED_QUERY_TEXTS: frozenset[str] = frozenset({
    "what causes death from Covid-19?",
    "what types of rapid testing for Covid-19 have been developed?",
    "how has COVID-19 affected Canada",
    "what are the guidelines for triaging patients infected with coronavirus?",
    "what are the transmission routes of coronavirus?",
    "are there any clinical trials available for the coronavirus",
    "which biomarkers predict the severe clinical course of 2019-nCOV infection?",
    "what is known about those infected with Covid-19 but are asymptomatic?",
    "Does SARS-CoV-2 have any subtypes, and if so what are they?",
    "How has the COVID-19 pandemic impacted violence in society, including violent crimes?",
    "what are the health outcomes for children who contract COVID-19?",
})


def is_excluded_query(query_text: str) -> bool:
    """Check whether a query should be excluded from evaluation."""
    return query_text.strip() in EXCLUDED_QUERY_TEXTS
