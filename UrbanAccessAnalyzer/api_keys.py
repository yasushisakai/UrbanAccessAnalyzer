"""API key loading with .env support.

Priority:
1) Real environment variables
2) .env file in current working directory
3) .env file at repository root (parent of UrbanAccessAnalyzer package)
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv_candidates() -> None:
    package_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / ".env",
        package_root / ".env",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_dotenv_file(resolved)


_load_dotenv_candidates()

# Tsunami notebook / batch flow
DEM = os.getenv("DEM") or os.getenv("DEM_API_KEY") or os.getenv("OPENTOPOGRAPHY_API_KEY", "")

# Optional census flows elsewhere in the package
US_CENSUS = os.getenv("US_CENSUS", "")
