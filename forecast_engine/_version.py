"""The engine's version, stamped per build.

A Databricks cluster installs a wheel with pip, and pip skips a package
whose version is already present. With a permanently static "0.1.0" that
made a long-lived all-purpose cluster keep whatever build it installed
first — forever. Every job cluster is new, so cloud runs on fresh compute
looked fine while the same code failed on Existing Compute against a
months-old wheel, with an argument-parser error from a CLI that had since
changed.

CI sets FORECAST_ENGINE_VERSION to a value carrying its run number, which
is a distinct PEP 440 local version ("0.1.0+ci.42"), so pip installs it
over whatever is there. A local build with the variable unset keeps the
plain base version, so nothing about developing here changes.
"""

from __future__ import annotations

import os

BASE_VERSION = "0.1.0"

__version__ = os.environ.get("FORECAST_ENGINE_VERSION", "").strip() or BASE_VERSION
