from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bet_slip_watcher.models import BetSlipReading
from bet_slip_watcher.state import wager_key


def _slip_ui_mask(bgr: np.ndarray) -> np.ndarray:
    """Bright slip modal + bet365 green (exclude empty letterbox)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = gray >= 90
    green = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 60)
        & (hsv[:, :, 2] >= 100)
    )
    return bright | green


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    if mask.size == 0 or not np.any(mask):
        return None
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [len(idx) - 1]))
    best = (0, 0, 0)
    for s, e in zip(starts, ends, strict=False):
        lo, hi = int(idx[s]), int(idx[e])
        run = hi - lo + 1
        if run > best[2]:
            best = (lo, hi, run)
    return best[0], best[1]


def tighten_slip_crop_margins(
    bgr: np.ndarray,
    *,
    pad: int = 12,
    min_col_density: float = 0.08,
    min_row_density: float = 0.05,
) -> np.ndarray:
    """
    Trim black letterbox left/right (and bottom chrome) around the slip modal.
    OCR uses the full config region; this is for review + Telegram only.
    """
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    ui = _slip_ui_mask(bgr)
    col_run = _largest_true_run(ui.mean(axis=0) >= min_col_density)
    if col_run is None:
        return bgr
    x0, x1 = col_run
    band = ui[:, x0 : x1 + 1]
    row_run = _largest_true_run(band.mean(axis=1) >= min_row_density)
    if row_run is None:
        y0, y1 = 0, h - 1
    else:
        y0, y1 = row_run
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return bgr
    return bgr[y0 : y1 + 1, x0 : x1 + 1]


def bet_slip_crop_for_export(
    frame: np.ndarray,
    regions,
    *,
    tighten: bool = True,
    pad: int = 12,
    min_col_density: float = 0.08,
) -> np.ndarray:
    crop = regions.bet_slip.crop(frame)
    if crop.size == 0:
        return crop
    if tighten:
        crop = tighten_slip_crop_margins(
            crop, pad=pad, min_col_density=min_col_density
        )
    return crop


def encode_crop_slip_png(
    frame: np.ndarray,
    regions,
    *,
    tighten: bool = True,
    pad: int = 12,
    min_col_density: float = 0.08,
) -> bytes | None:
    """Same content as review ``*_crop-slip.png`` (optionally margin-trimmed)."""
    crop = bet_slip_crop_for_export(
        frame,
        regions,
        tighten=tighten,
        pad=pad,
        min_col_density=min_col_density,
    )
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return buf.tobytes()


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


def ocr_empty_for_review(reading: BetSlipReading) -> bool:
    """True when Telegram would show the tune-regions placeholder."""
    if not reading.summary_lines():
        return True
    return not wager_key(reading).replace("|", "").strip()


def review_capture_stem(
    frame_idx: int,
    reading: BetSlipReading,
    *,
    event_name: str,
    ocr_empty: bool,
) -> str:
    ref = _safe_token(reading.bet_ref or "no-ref", 24)
    stake = int(reading.stake_rsd) if reading.stake_rsd else 0
    parts = [f"{frame_idx:06d}", "NOTIFY", _safe_token(event_name, 24)]
    if ocr_empty:
        parts.append("OCR-EMPTY")
    parts.extend([f"stake{stake}", f"ref-{ref}"])
    return "_".join(parts)


def save_review_capture(
    directory: Path,
    frame: np.ndarray,
    frame_idx: int,
    reading: BetSlipReading,
    *,
    event_name: str,
    regions,
    tighten_slip: bool = True,
    tighten_pad: int = 12,
    tighten_min_col: float = 0.08,
) -> Path:
    """Full-frame screenshot (+ region crops) for the analyzed frame at notify time."""
    directory.mkdir(parents=True, exist_ok=True)
    empty = ocr_empty_for_review(reading)
    stem = review_capture_stem(
        frame_idx, reading, event_name=event_name, ocr_empty=empty
    )
    full_path = directory / f"{stem}.png"
    cv2.imwrite(str(full_path), frame)
    if regions is not None:
        slip_export = bet_slip_crop_for_export(
            frame,
            regions,
            tighten=tighten_slip,
            pad=tighten_pad,
            min_col_density=tighten_min_col,
        )
        cv2.imwrite(str(directory / f"{stem}_crop-slip.png"), slip_export)
        cv2.imwrite(
            str(directory / f"{stem}_crop-search.png"), regions.search_bar.crop(frame)
        )
        cv2.imwrite(
            str(directory / f"{stem}_crop-market.png"), regions.market_row.crop(frame)
        )
    return full_path


def save_timeline_snapshot(
    directory: Path,
    frame: np.ndarray,
    frame_idx: int,
    *,
    elapsed_s: float,
    placed_banner: bool,
    bet_placed: bool,
    bet_placing: bool,
    awaiting_click: bool,
    has_latch: bool,
    btn_visible: bool,
    submit_started: bool,
) -> Path:
    """Periodic full-frame capture for debugging detection lag."""
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = (
        f"banner-{int(placed_banner)}"
        f"_placed-{int(bet_placed)}"
        f"_placing-{int(bet_placing)}"
        f"_await-{int(awaiting_click)}"
        f"_latch-{int(has_latch)}"
        f"_btn-{int(btn_visible)}"
        f"_submit-{int(submit_started)}"
    )
    name = f"{ts}_t{int(elapsed_s):05d}_f{frame_idx:06d}_{flags}.png"
    path = directory / name
    cv2.imwrite(str(path), frame)
    return path
