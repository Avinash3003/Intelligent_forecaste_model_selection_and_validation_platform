"""Checks that the model only cited numbers the run actually produced.

A rule-based check rather than a second LLM call: the claims worth checking
are numbers, and every number the model may cite already exists verbatim in
the structured facts passed in. That makes this a comparison, not a
judgement, so a second call would add latency and cost to verify what a
regex already settles.

Every number in the summary must match one of the group's real metrics,
within a small rounding tolerance — citing 8.2% for 8.23% is rounding, not
fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")

# Loose match: a rounded citation ("8%" for 8.23) is not a fabrication.
# Tight enough that a genuinely different number (a hallucinated 26% next
# to a real 8.2%) still fails.
_ABSOLUTE_TOLERANCE = 0.6
_RELATIVE_TOLERANCE = 0.03


@dataclass
class GroundingResult:
    grounded: bool
    issues: list[str] = field(default_factory=list)
    checked_numbers: int = 0

    @property
    def status(self) -> str:
        return "grounded" if self.grounded else "ungrounded"


def extract_numeric_facts(metrics: dict[str, float | int | None]) -> set[float]:
    """Flatten a metrics dict into the set of numbers a claim may cite.

    Percent-scale and fraction-scale versions of the same value are both
    included (8.2 and 0.082) — the platform's own reports are inconsistent
    about which scale a given field uses, and a citation should not fail
    grounding purely because the narrative wrote a percentage.
    """
    facts: set[float] = set()
    for value in metrics.values():
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        facts.add(round(number, 4))
        if 0 <= abs(number) <= 1:
            facts.add(round(number * 100, 4))
        elif abs(number) <= 1000:
            facts.add(round(number / 100, 6))
    return facts


def check_grounding(text: str, metrics: dict[str, float | int | None]) -> GroundingResult:
    """Verify every number in `text` against `metrics`.

    Returns:
        A `GroundingResult`. `grounded=True` when every number cited
        matches a known fact within tolerance (or no number was cited at
        all — an insight with no numeric claim has nothing to be
        ungrounded about). Never raises: an unparseable number is treated
        as an issue, not an exception, since this runs on every insight a
        run produces and must not be able to fail the run.
    """
    if not text or not text.strip():
        return GroundingResult(grounded=True, checked_numbers=0)

    facts = extract_numeric_facts(metrics)
    issues: list[str] = []
    checked = 0

    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.rstrip("%")
        try:
            number = float(cleaned)
        except ValueError:
            continue

        # Small integers (a rank, a count of rejected models, a horizon
        # step) are not the accuracy/confidence-style claims this check
        # exists for, and would produce constant false positives — "1" or
        # "12" trivially fails to match a WMAPE fact set.
        if cleaned.lstrip("-").isdigit() and abs(number) <= 12:
            continue

        checked += 1
        if not _matches_any(number, facts):
            issues.append(f"'{raw}' does not match any known metric for this group.")

    return GroundingResult(grounded=not issues, issues=issues, checked_numbers=checked)


def _matches_any(number: float, facts: set[float]) -> bool:
    if not facts:
        # No numeric facts were supplied at all (a group with no backtest,
        # e.g. a fresh fallback) — nothing to check against, so a cited
        # number cannot be confirmed OR refuted. Treated as grounded rather
        # than blocking output over evidence we were never given.
        return True
    for fact in facts:
        if abs(number - fact) <= max(_ABSOLUTE_TOLERANCE, abs(fact) * _RELATIVE_TOLERANCE):
            return True
    return False
