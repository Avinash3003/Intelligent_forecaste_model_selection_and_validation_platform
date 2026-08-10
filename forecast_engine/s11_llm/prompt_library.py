"""Prompt loading (Section 6.12, "Prompt Architecture") + versioning
(Section 13.1, "Prompt versioning as a first-class artifact").

Prompts are plain text files under `forecast_engine/s11_llm/prompts/<version>/`,
never strings embedded in business logic — editing a prompt's wording, or
adding a new one, is a file change, not a code change.

Versioning is a directory per version rather than a filename suffix
(`structured_insight_v2.txt`) because it makes "what does v1 look like next
to v2" a `diff -r prompts/v1 prompts/v2`, and because a version is a
complete, self-contained set of templates — a system prompt written for
free-text narrative and one written for structured JSON output do not mix
across a version boundary, so nothing here lets a caller combine a system
prompt from one version with a template from another.

`string.Template`'s `$name` substitution (rather than `str.format`) is used
deliberately: `safe_substitute` leaves an unrecognized placeholder
untouched instead of raising, so a template can be edited to reference a
new variable without every call site needing to supply it immediately.
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

# The active structured-insight contract (Section 13.1). "v1" is the
# retired free-text, whole-run engine's prompt set, kept on disk as a
# historical record of what shipped before this phase — see
# `prompts/v1/` — but no code path in this engine calls it any more, since
# Section 1.2 requires structured JSON to be the primary contract, not an
# alternative one. "v2" is a live target: `LLMConfig.prompt_version`
# selects it, and a future v3 rolls forward the same way.
STRUCTURED_INSIGHT_TEMPLATE = "structured_insight"


class PromptLibrary:
    """Loads and fills the prompt templates for one prompt version."""

    # Initialize the template cache and prompts directory
    def __init__(self, prompts_dir: Path | None = None, version: str = "v2") -> None:
        self._root = prompts_dir or _PROMPTS_DIR
        self._version = version
        self._cache: dict[str, string.Template] = {}

    @property
    def version(self) -> str:
        return self._version

    # Load the shared system prompt template for this version
    def system_prompt(self) -> str:
        return self._load("system_prompt").template

    # Fill template `name` (this version's) with `variables`
    def render(self, name: str, variables: dict[str, Any]) -> str:
        template = self._load(name)
        return template.safe_substitute(**{key: _stringify(value) for key, value in variables.items()})

    # Every prompt version present on disk, for a version-comparison view
    def available_versions(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(
            child.name
            for child in self._root.iterdir()
            if child.is_dir() and (child / "system_prompt.txt").is_file()
        )

    # This version's raw template source, for a side-by-side diff/rollback
    # decision — reads directly rather than through the cache, since a
    # comparison view should never be affected by what has been rendered
    # in this process already.
    def raw_template(self, name: str, version: str | None = None) -> str:
        path = self._root / (version or self._version) / f"{name}.txt"
        if not path.exists():
            raise LLMProviderError(f"Prompt template '{name}' was not found at {path}.")
        return path.read_text()

    # Load and cache a template file by name, scoped to this version
    def _load(self, name: str) -> string.Template:
        cache_key = f"{self._version}/{name}"
        if cache_key not in self._cache:
            path = self._root / self._version / f"{name}.txt"
            if not path.exists():
                raise LLMProviderError(
                    f"Prompt template '{name}' (version '{self._version}') was not found at {path}."
                )
            self._cache[cache_key] = string.Template(path.read_text())
        return self._cache[cache_key]


# Coerce a variable to its string form for substitution
def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
