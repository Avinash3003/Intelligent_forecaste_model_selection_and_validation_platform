"""The JSON contract every LLM insight must satisfy, plus its validator.

Free-text prose is fragile to grade and audit, so the model returns JSON
against a fixed schema and this checks it before anything reaches the
dashboard.

Plain dataclasses rather than pydantic, matching the rest of the engine,
which has no pydantic dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Hard caps mirrored from the prompt itself (Section 13/UI requirement:
# "approximately 2-5 short sentences"). Enforced here too — a prompt is a
# request, this is the guarantee. Matches backend/app/schemas/results.py's
# DashboardInsight caps so the two layers agree on what "concise" means.
MAX_SUMMARY_WORDS = 70
MAX_REJECTION_REASONS = 6
MAX_CAVEATS = 3
MAX_REASON_WORDS = 20
MAX_CAVEAT_WORDS = 20


class SchemaValidationError(Exception):
    """The raw LLM response does not satisfy `InsightPayload`'s contract.

    Carries a list of specific, human-readable problems — this is what
    gets fed back into a retry prompt (Section 13.1: "retry mechanism with
    failure insight embedded back into the prompt"), so the model sees
    exactly what was wrong rather than a bare "invalid, try again".
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class InsightPayload:
    """One forecast group's structured explanation — the primary output
    contract Section 13.1 requires, replacing free-text narrative.
    """

    selected_model: str
    rejection_reasons: list[str] = field(default_factory=list)
    confidence: float | None = None
    caveats: list[str] = field(default_factory=list)
    concise_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "rejection_reasons": list(self.rejection_reasons),
            "confidence": self.confidence,
            "caveats": list(self.caveats),
            "concise_summary": self.concise_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InsightPayload":
        return cls(
            selected_model=str(payload.get("selected_model") or ""),
            rejection_reasons=[str(r) for r in (payload.get("rejection_reasons") or [])],
            confidence=_coerce_float(payload.get("confidence")),
            caveats=[str(c) for c in (payload.get("caveats") or [])],
            concise_summary=str(payload.get("concise_summary") or ""),
        )


def parse_and_validate(raw_text: str, *, expected_model: str | None = None) -> InsightPayload:
    """Parse `raw_text` as JSON and validate it against `InsightPayload`'s
    contract.

    Raises:
        SchemaValidationError: with every problem found, not just the
            first — so a retry prompt can address all of them at once
            rather than discovering them one failed attempt at a time.
    """
    problems: list[str] = []

    stripped = _strip_code_fence(raw_text)
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SchemaValidationError([f"Response is not valid JSON: {exc}"]) from exc

    if not isinstance(payload, dict):
        raise SchemaValidationError(["Response JSON must be an object, not a list or scalar."])

    required = ("selected_model", "rejection_reasons", "confidence", "caveats", "concise_summary")
    missing = [key for key in required if key not in payload]
    if missing:
        problems.append(f"Missing required field(s): {', '.join(missing)}.")

    selected_model = payload.get("selected_model")
    if not isinstance(selected_model, str) or not selected_model.strip():
        problems.append("'selected_model' must be a non-empty string.")
    elif expected_model and selected_model.strip().lower() != expected_model.strip().lower():
        problems.append(
            f"'selected_model' is '{selected_model}' but the context names '{expected_model}' as the "
            "actual selected model."
        )

    rejection_reasons = payload.get("rejection_reasons")
    if rejection_reasons is not None and not _is_str_list(rejection_reasons):
        problems.append("'rejection_reasons' must be a list of strings.")
    elif isinstance(rejection_reasons, list):
        if len(rejection_reasons) > MAX_REJECTION_REASONS:
            problems.append(f"'rejection_reasons' has {len(rejection_reasons)} entries; at most {MAX_REJECTION_REASONS} allowed.")
        for reason in rejection_reasons:
            if len(str(reason).split()) > MAX_REASON_WORDS:
                problems.append(f"A rejection reason exceeds {MAX_REASON_WORDS} words: '{reason[:60]}...'.")
                break

    confidence = payload.get("confidence")
    if confidence is not None:
        numeric = _coerce_float(confidence)
        if numeric is None:
            problems.append("'confidence' must be a number or null.")
        elif not (0 <= numeric <= 100):
            problems.append(f"'confidence' must be between 0 and 100; got {numeric}.")

    caveats = payload.get("caveats")
    if caveats is not None and not _is_str_list(caveats):
        problems.append("'caveats' must be a list of strings.")
    elif isinstance(caveats, list):
        if len(caveats) > MAX_CAVEATS:
            problems.append(f"'caveats' has {len(caveats)} entries; at most {MAX_CAVEATS} allowed.")
        for caveat in caveats:
            if len(str(caveat).split()) > MAX_CAVEAT_WORDS:
                problems.append(f"A caveat exceeds {MAX_CAVEAT_WORDS} words: '{caveat[:60]}...'.")
                break

    summary = payload.get("concise_summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("'concise_summary' must be a non-empty string.")
    elif len(summary.split()) > MAX_SUMMARY_WORDS:
        problems.append(
            f"'concise_summary' is {len(summary.split())} words; the UI requires at most "
            f"{MAX_SUMMARY_WORDS} words (2-5 short sentences)."
        )

    if problems:
        raise SchemaValidationError(problems)

    return InsightPayload.from_dict(payload)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` fence if the model wrapped its JSON in one.

    The prompt asks for bare JSON, but models routinely wrap it in a
    markdown fence anyway; stripping it here is more reliable than asking
    harder, and keeps `parse_and_validate` from failing on formatting alone
    when the JSON itself is otherwise perfectly valid.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
