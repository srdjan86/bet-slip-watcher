from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GreenButtonMetrics:
    ratio: float
    visible: bool


def place_bet_green_ratio(slip_bgr: np.ndarray, hsv_cfg: dict) -> GreenButtonMetrics:
    if slip_bgr.size == 0:
        return GreenButtonMetrics(ratio=0.0, visible=False)

    hsv = cv2.cvtColor(slip_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [hsv_cfg["h_min"], hsv_cfg["s_min"], hsv_cfg["v_min"]], dtype=np.uint8
    )
    upper = np.array([hsv_cfg["h_max"], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    visible = ratio >= float(hsv_cfg.get("min_green_pixel_ratio", 0.01))
    return GreenButtonMetrics(ratio=ratio, visible=visible)


def place_bet_button_metrics(slip_bgr: np.ndarray, hsv_cfg: dict) -> GreenButtonMetrics:
    """Green on the bottom action bar (Place Bet, spinner, or Bet Placed)."""
    if slip_bgr.size == 0:
        return GreenButtonMetrics(ratio=0.0, visible=False)
    h, w = slip_bgr.shape[:2]
    btn = slip_bgr[int(h * 0.52) :, int(w * 0.32) :]
    return place_bet_green_ratio(btn, hsv_cfg)


def action_bar_crop(slip_bgr: np.ndarray) -> np.ndarray:
    if slip_bgr.size == 0:
        return slip_bgr
    h, w = slip_bgr.shape[:2]
    return slip_bgr[int(h * 0.45) :, int(w * 0.32) :]


def action_bar_motion_sample(slip_bgr: np.ndarray) -> np.ndarray | None:
    """Small grayscale patch for detecting Place Bet → spinner animation."""
    bar = action_bar_crop(slip_bgr)
    if bar.size == 0:
        return None
    gray = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (48, 24), interpolation=cv2.INTER_AREA)


def action_bar_white_on_green_ratio(slip_bgr: np.ndarray, hsv_cfg: dict) -> float:
    """White pixels on the green action bar (spinner arc vs dark Place Bet label)."""
    bar = action_bar_crop(slip_bgr)
    if bar.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [hsv_cfg["h_min"], hsv_cfg["s_min"], hsv_cfg["v_min"]], dtype=np.uint8
    )
    upper = np.array([hsv_cfg["h_max"], 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 100, 255))
    white_on_green = cv2.bitwise_and(white, green)
    green_px = max(1, int(np.count_nonzero(green)))
    return float(np.count_nonzero(white_on_green)) / float(green_px)


def place_bet_cta_visible(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    peak_wog: float,
    *,
    spinner_drop_ratio: float = 0.45,
) -> bool:
    """
    Idle Place Bet button (user has not clicked yet).
    After click, white-on-green ratio drops sharply (spinner / processing).
    """
    btn = place_bet_button_metrics(slip_bgr, hsv_cfg)
    if not btn.visible:
        return False
    wog = action_bar_white_on_green_ratio(slip_bgr, hsv_cfg)
    if peak_wog <= 0:
        return True
    return wog >= peak_wog * spinner_drop_ratio


def action_bar_motion(prev: np.ndarray | None, curr: np.ndarray | None) -> float:
    if prev is None or curr is None or prev.shape != curr.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(prev, curr)))


def bet_placing_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    panel: float,
    place_bet_visible: bool,
    bet_placed_confirmed: bool,
    stake_rsd: float | None,
) -> bool:
    """
    Spinner / processing state: stake on slip, green action button, no Place Bet label yet.
    """
    if bet_placed_confirmed:
        return False
    if place_bet_visible:
        return False
    if panel < 0.45:
        return False
    btn = place_bet_button_metrics(slip_bgr, hsv_cfg)
    if not btn.visible:
        return False
    if stake_rsd is not None and stake_rsd > 0:
        return True
    # Spinner / processing: green bar without "Place Bet" OCR (stake may fail OCR)
    return True


def slip_ui_busy(
    panel: float,
    green: GreenButtonMetrics,
    btn: GreenButtonMetrics,
    *,
    min_panel: float = 0.22,
) -> bool:
    """
    Cheap gate before Tesseract. Panel score alone is noisy (dark page in crop);
    require green on the slip or action bar.
    """
    del min_panel  # kept in config/API for tuning experiments
    return green.visible or btn.visible


def slip_panel_score(slip_bgr: np.ndarray) -> float:
    """Heuristic: bet slip is a dark bar; score mean darkness in crop."""
    if slip_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(slip_bgr, cv2.COLOR_BGR2GRAY)
    # Dark UI ~ 20-60 on bet365 slip background
    dark = np.mean(gray < 90)
    return float(dark)


def wager_outcome_ui_active(
    panel: float,
    green: GreenButtonMetrics,
    btn: GreenButtonMetrics,
    placed_banner: bool,
    *,
    min_panel: float = 0.45,
) -> bool:
    """
    Post-submit slip UI (spinner or Bet Placed header), cheap — no OCR.
    Green on slip + panel, or Bet Placed banner in the top strip.
    """
    if placed_banner:
        return True
    if panel < min_panel:
        return False
    return green.visible or btn.visible


def bet_placed_banner_heuristic(slip_bgr: np.ndarray) -> bool:
    """
    Detect the light "Bet Placed" title band without Tesseract.
    Tuned on bet365 slip crops (1080p desktop stream).
    """
    if slip_bgr.size == 0:
        return False
    h, w = slip_bgr.shape[:2]
    strip = slip_bgr[: int(h * 0.22), int(w * 0.05) : int(w * 0.95)]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark = float(np.mean(gray < 85))
    bright = float(np.mean(gray > 175))
    return dark >= 0.55 and bright >= 0.008
