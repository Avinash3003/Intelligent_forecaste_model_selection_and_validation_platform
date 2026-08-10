"""Prompt injection defense (Section 13.1, "Guardrails for injection/leakage").

Every value in the structured metrics payload the engine builds internally
— WMAPE, drift statistics, ranking scores — is trusted: it was computed by
this pipeline, from this pipeline's own data. It never needs sanitizing.

What is *not* trusted is anything a user typed into the wizard's column
mapping: a date column, target column, key column or feature column name.
Those strings flow into the prompt verbatim today (`context_formatter.py`'s
dataset section), and nothing stops a user from naming a column

    "sales\\n\\nIGNORE ALL PREVIOUS INSTRUCTIONS AND ..."

A column name is never validated against the dataset for *safety* — only
for existence (`ForecastConfiguration.validate_against_columns`) — so this
is the one place user text is neutralized before it can reach a prompt.
"""

from __future__ import annotations

import re
import unicodedata

# A generous but bounded length: real column names are short, and an
# injection attempt often relies on a very long payload to push the
# instruction far enough from its context to read as "new" content.
MAX_FIELD_LENGTH = 120

# Phrases that only make sense as an attempt to redirect the model, never
# as part of a legitimate column name. Matched case-insensitively; each
# match is stripped, not just flagged, so a name can be sanitized and still
# used.
_INSTRUCTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\b",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"\bact\s+as\b",
    r"\breveal\s+(the\s+)?(system\s+)?prompt\b",
    r"\bprint\s+(the\s+)?(system\s+)?prompt\b",
)
_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)

# Markdown/structural tokens that let injected text imitate a new prompt
# section (the same "## Heading" convention `context_formatter.py` uses for
# its own trusted sections) or a role-switch marker.
_STRUCTURAL_RE = re.compile(r"[#`]{2,}|^\s*(system|assistant|user)\s*:", re.IGNORECASE | re.MULTILINE)

# Control characters (other than plain whitespace) have no legitimate
# reason to appear in a column name and are sometimes used to obscure text
# from a naive filter.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_for_prompt(value: str | None) -> str:
    """Neutralize `value` for safe inclusion in an LLM prompt.

    Applied to every piece of user-supplied metadata (column names) before
    `context_formatter` renders it into a prompt. Never raises — a field
    that sanitizes down to nothing becomes an empty string, and the caller
    decides how to handle that, exactly like any other missing value.
    """
    if not value:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_RE.sub(" ", text)
    text = _INSTRUCTION_RE.sub("[redacted]", text)
    text = _STRUCTURAL_RE.sub(" ", text)

    # Collapse newlines: a column name spanning multiple lines is either a
    # data-entry mistake or an attempt to open new prompt "sections" via
    # blank-line separation, and neither is legitimate here.
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_FIELD_LENGTH:
        text = text[:MAX_FIELD_LENGTH].rstrip() + "…"

    return text


def sanitize_columns(columns: list[str] | tuple[str, ...] | None) -> list[str]:
    """`sanitize_for_prompt` applied to a whole column list, in order."""
    if not columns:
        return []
    return [sanitize_for_prompt(column) for column in columns]
