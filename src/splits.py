"""Chronological train/validation/test guards.

The test period is deliberately unavailable unless the caller explicitly sets
``final=True`` (or passes ``--final`` in a command-line entry point).
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as cfg


def _utc_boundary(value: str) -> pd.Timestamp:
    """Return the UTC instant at local midnight for a split boundary."""
    return pd.Timestamp(value).tz_localize(cfg.LOCAL_TZ).tz_convert("UTC")


def split_mask(ts_utc: pd.Series, split: str, *, final: bool = False) -> pd.Series:
    """Return a boolean chronological split mask; gate test rows behind ``final``."""
    if not isinstance(ts_utc.dtype, pd.DatetimeTZDtype):
        ts_utc = pd.to_datetime(ts_utc, utc=True)
    boundaries = {
        "train": (cfg.TRAIN_START, cfg.TRAIN_END),
        "validation": (cfg.VALID_START, cfg.VALID_END),
        "test": (cfg.TEST_START, None),
    }
    if split not in boundaries:
        raise ValueError(f"unknown split {split!r}; choose {sorted(boundaries)}")
    if split == "test" and not final:
        raise PermissionError("The 2026 test split is locked. Re-run explicitly with --final.")
    start, end = boundaries[split]
    mask = ts_utc >= _utc_boundary(start)
    if end:
        mask &= ts_utc < _utc_boundary(end)
    return mask


def add_final_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final", action="store_true", help="unlock the held-out 2026 test split")
