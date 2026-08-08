"""Prompt loading (Section 6.12, "Prompt Architecture").

Prompts are plain text files under `forecast_engine/llm/prompts/`, never
strings embedded in business logic — editing a prompt's wording, or adding
a new one, is a file change, not a code change. `string.Template`'s
`$name` substitution (rather than `str.format`) is used deliberately:
`safe_substitute` leaves an unrecognized placeholder untouched instead of
raising, so a template can be edited to reference a new variable without
every call site needing to supply it immediately.
"""

from __future__ import annotations

import string
from importlib import resources
from pathlib import Path
from typing import Any

from forecast_engine.utils.exceptions import LLMProviderError


def _default_prompts_dir() -> Path:
    # Resolved through the import system (`importlib.resources`) rather
    # than assuming `__file__` sits on a real filesystem path next to this
    # module — the portable way to reach packaged data whether the package
    # was installed from a wheel, an editable install, or a Databricks
    # workspace deployment, none of which are guaranteed to lay out files
    # exactly like a repo checkout.
    return Path(str(resources.files("forecast_engine.s11_llm") / "prompts"))


_PROMPTS_DIR = _default_prompts_dir()

# The complete set of insight sections this phase produces (Section 6.12's
# OUTPUT list). Adding an eighth section is adding a `.txt` file here plus
# one entry in this tuple — `LLMInsightEngine` iterates it generically.
INSIGHT_SECTIONS: tuple[str, ...] = (
    "technical_explanation",
    "business_explanation",
    "forecast_summary",
    "winner_model_summary",
    "important_features",
    "forecast_risks",
    "drift_summary",
)


class PromptLibrary:
    """Loads and fills the prompt templates backing each insight section."""

    # Initialize the template cache and prompts directory
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or _PROMPTS_DIR
        self._cache: dict[str, string.Template] = {}

    # Load the shared system prompt template
    def system_prompt(self) -> str:
        return self._load("system_prompt").template

    # Fill template `name` with `variables`
    def render(self, name: str, variables: dict[str, Any]) -> str:
        template = self._load(name)
        return template.safe_substitute(**{key: _stringify(value) for key, value in variables.items()})

    # Load and cache a template file by name
    def _load(self, name: str) -> string.Template:
        if name not in self._cache:
            path = self._dir / f"{name}.txt"
            if not path.exists():
                raise LLMProviderError(f"Prompt template '{name}' was not found at {path}.")
            self._cache[name] = string.Template(path.read_text())
        return self._cache[name]


# Coerce a variable to its string form for substitution
def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
