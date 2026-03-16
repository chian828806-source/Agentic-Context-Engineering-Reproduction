"""
Lightweight .env loader with optional python-dotenv support.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """
    Load environment variables from .env if available.

    Prefers python-dotenv when installed; falls back to a minimal parser.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except Exception:
        pass

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
