from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from bet_slip_watcher.models import BetSlipReading


def _safe_token(text: str, max_len: int = 32) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return cleaned[:max_len] or "unknown"


def audit_frame_name(
    frame_idx: int,
    reading: BetSlipReading,
    slip_visible: bool,
    *,
    prefix: str = "",
) -> str:
    placed = "bet-placed" if reading.bet_placed_confirmed else "not-placed"
    placing = "bet-placing" if reading.bet_placing_active else "not-placing"
    ready = (
        "slip-ready"
        if reading.place_bet_visible and reading.stake_rsd
        else "not-ready"
    )
    slip = "slip-on-screen" if slip_visible else "no-slip"
    stake = int(reading.stake_rsd) if reading.stake_rsd else 0
    ref = _safe_token(reading.bet_ref or "no-ref", 24)
    parts = [
        f"{frame_idx:06d}",
        placed,
        placing,
        ready,
        slip,
        f"stake{stake}",
        f"ref-{ref}",
    ]
    if prefix:
        parts.insert(0, _safe_token(prefix, 40))
    return "_".join(parts) + ".png"


def save_audit_frame(
    directory: Path,
    frame: np.ndarray,
    frame_idx: int,
    reading: BetSlipReading,
    slip_visible: bool,
    *,
    prefix: str = "",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / audit_frame_name(
        frame_idx, reading, slip_visible, prefix=prefix
    )
    cv2.imwrite(str(path), frame)
    return path
