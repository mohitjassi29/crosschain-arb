"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from crosschain_arb.config import RESULTS_DIR


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    return logging.getLogger("ccarb")


def write_json(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path
