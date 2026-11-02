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
    *,
    settled_history: bool = False,
) -> bool:
    """
    Spinner / processing state on the compact bet slip modal (green bar, no Place Bet).
    """
    if bet_placed_confirmed or place_bet_visible or settled_history:
        return False
    if bet_slip_spinner_heuristic(
        slip_bgr,
        hsv_cfg,
        panel=panel,
        settled_history=settled_history,
        place_bet_visible=False,
    ):
        return True
    if panel < 0.45:
        return False
    if not compact_wager_modal_heuristic(
        slip_bgr, hsv_cfg, panel=panel, settled_history=settled_history
    ):
        return False
    btn = place_bet_button_metrics(slip_bgr, hsv_cfg)
    if not btn.visible:
        return False
    if stake_rsd is not None and stake_rsd > 0:
        return True
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


def bet_placed_green_header(slip_bgr: np.ndarray, hsv_cfg: dict) -> bool:
    """Green title bar with checkmark (Bet Placed header on bet365)."""
    if slip_bgr.size == 0:
        return False
    h, w = slip_bgr.shape[:2]
    top = slip_bgr[: max(8, int(h * 0.18)), int(w * 0.02) : int(w * 0.98)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    lo = (int(hsv_cfg["h_min"]), max(40, int(hsv_cfg["s_min"]) // 2), int(hsv_cfg["v_min"]))
    hi = (int(hsv_cfg["h_max"]), 255, 255)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    return float(np.mean(mask)) / 255.0 >= 0.045


def wager_modal_light_body_ratio(slip_bgr: np.ndarray) -> float:
    """White card body in the compact bet slip / Bet Placed popup (not dark page fill)."""
    if slip_bgr.size == 0:
        return 0.0
    h, w = slip_bgr.shape[:2]
    mid = slip_bgr[int(h * 0.20) : int(h * 0.58), int(w * 0.08) : int(w * 0.92)]
    gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 110))


def compact_wager_modal_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    panel: float,
    settled_history: bool = False,
) -> bool:
    """
    Floating bet slip modal (Place Bet / spinner / Bet Placed), not full-page My Bets or lobby.
    """
    if settled_history or slip_bgr.size == 0:
        return False
    if panel < 0.28 or panel > 0.93:
        return False
    if bet_placed_green_header(slip_bgr, hsv_cfg):
        return True
    if wager_modal_light_body_ratio(slip_bgr) >= 0.12:
        return True
    btn = place_bet_button_metrics(slip_bgr, hsv_cfg)
    return btn.visible and panel <= 0.90


def bet_placed_checkmark_in_header(slip_bgr: np.ndarray, hsv_cfg: dict) -> bool:
    """White checkmark disc on the left of the green Bet Placed title bar."""
    if not bet_placed_green_header(slip_bgr, hsv_cfg):
        return False
    h, w = slip_bgr.shape[:2]
    patch = slip_bgr[: max(8, int(h * 0.17)), int(w * 0.02) : int(w * 0.13)]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 195)) >= 0.03


def bet_placed_banner_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict | None = None,
    *,
    panel: float = 0.0,
    settled_history: bool = False,
) -> bool:
    """
    Bet Placed popup: green title bar (checkmark on the left) on a compact white modal.
    Dark full-page crops must not match.
    """
    if slip_bgr.size == 0 or settled_history:
        return False
    if hsv_cfg and bet_placed_green_header(slip_bgr, hsv_cfg):
        return compact_wager_modal_heuristic(
            slip_bgr, hsv_cfg, panel=panel, settled_history=False
        )
    if panel >= 0.88 or not compact_wager_modal_heuristic(
        slip_bgr, hsv_cfg or {}, panel=panel, settled_history=False
    ):
        return False
    if wager_modal_light_body_ratio(slip_bgr) < 0.15:
        return False
    h, w = slip_bgr.shape[:2]
    strip = slip_bgr[: int(h * 0.22), int(w * 0.05) : int(w * 0.95)]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark = float(np.mean(gray < 85))
    bright = float(np.mean(gray > 175))
    if dark > 0.968:
        return False
    return dark >= 0.50 and bright >= 0.006


def bet_slip_spinner_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    panel: float,
    settled_history: bool,
    place_bet_visible: bool,
    max_white_on_green: float = 0.12,
) -> bool:
    """Green action bar with circular progress (no Place Bet label)."""
    if settled_history or place_bet_visible:
        return False
    if not compact_wager_modal_heuristic(
        slip_bgr, hsv_cfg, panel=panel, settled_history=False
    ):
        return False
    btn = place_bet_button_metrics(slip_bgr, hsv_cfg)
    if not btn.visible:
        return False
    wog = action_bar_white_on_green_ratio(slip_bgr, hsv_cfg)
    return wog <= max_white_on_green


def _slip_green_ratio(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    y0_frac: float,
    y1_frac: float,
    *,
    x0_frac: float = 0.05,
    x1_frac: float = 0.95,
) -> float:
    if slip_bgr.size == 0:
        return 0.0
    h, w = slip_bgr.shape[:2]
    y0 = max(0, int(h * y0_frac))
    y1 = max(y0 + 1, int(h * y1_frac))
    x0 = max(0, int(w * x0_frac))
    x1 = max(x0 + 1, int(w * x1_frac))
    band = slip_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    lo = (
        int(hsv_cfg["h_min"]),
        max(40, int(hsv_cfg["s_min"]) // 2),
        int(hsv_cfg["v_min"]),
    )
    hi = (int(hsv_cfg["h_max"]), 255, 255)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    return float(np.mean(mask)) / 255.0


def my_bets_list_in_slip_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    panel: float,
    min_panel: float = 0.85,
) -> bool:
    """
    My Bets (Open/Live/Settled) cards in the slip crop — WON pill / stake header green
    at the top, or Returned buttons (not Bet Placed title bar or Place Bet bar only).
    """
    if slip_bgr.size == 0 or panel < min_panel:
        return False
    if bet_placed_green_header(slip_bgr, hsv_cfg):
        return False
    top_green = _slip_green_ratio(slip_bgr, hsv_cfg, 0.0, 0.14)
    if top_green >= 0.038:
        return True
    upper = _slip_green_ratio(slip_bgr, hsv_cfg, 0.05, 0.42)
    lower = _slip_green_ratio(slip_bgr, hsv_cfg, 0.55, 0.98)
    if panel >= 0.90 and upper >= 0.012 and lower >= 0.04:
        return True
    return False


def fullpage_in_slip_crop_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    panel: float,
    min_panel: float = 0.94,
) -> bool:
    """
    bet365 homepage / My Bets / search fills the slip crop (dark, no action bar).
    Not the compact Place Bet / Bet Placed popup (green header or bottom bar).
    """
    if slip_bgr.size == 0 or panel < min_panel:
        return False
    if bet_placed_green_header(slip_bgr, hsv_cfg):
        return False
    if place_bet_button_metrics(slip_bgr, hsv_cfg).visible:
        return False
    return True


def bet_settled_history_heuristic(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    panel: float = 0.0,
) -> bool:
    """
    My Bets (Open/Live/Settled) list in the slip crop — not Place Bet / Bet Placed popup.
    """
    if slip_bgr.size == 0:
        return False
    bands_strict = _count_green_stake_bands(slip_bgr, hsv_cfg, min_ratio=0.06)
    if bands_strict >= 3:
        return True
    if panel >= 0.90:
        bands_relaxed = _count_green_stake_bands(slip_bgr, hsv_cfg, min_ratio=0.04)
        if bands_relaxed >= 2:
            return True
    return False


def _count_green_stake_bands(
    slip_bgr: np.ndarray,
    hsv_cfg: dict,
    *,
    min_ratio: float,
) -> int:
    h, w = slip_bgr.shape[:2]
    hsv = cv2.cvtColor(slip_bgr, cv2.COLOR_BGR2HSV)
    lo = (
        int(hsv_cfg["h_min"]),
        max(40, int(hsv_cfg["s_min"]) // 2),
        int(hsv_cfg["v_min"]),
    )
    hi = (int(hsv_cfg["h_max"]), 255, 255)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    band_h = max(4, int(h * 0.07))
    green_bands = 0
    step = max(1, int(h * 0.12))
    for y in range(int(h * 0.08), int(h * 0.85), step):
        band = mask[y : y + band_h, int(w * 0.05) : int(w * 0.95)]
        if band.size and float(np.mean(band)) / 255.0 >= min_ratio:
            green_bands += 1
    return green_bands
