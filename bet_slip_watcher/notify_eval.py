"""Evaluate whether a screenshot would trigger bet_placing / bet_placed notifications."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

from bet_slip_watcher.detect import (
    bet_placed_green_header,
    bet_placed_modal_detected,
    bet_placing_heuristic,
    bet_settled_history_heuristic,
    fullpage_in_slip_crop_heuristic,
    my_bets_list_in_slip_heuristic,
    non_wager_popup_in_slip,
    place_bet_button_metrics,
    place_bet_green_ratio,
    place_bet_modal_idle,
    slip_panel_score,
    sports_markets_page_in_slip,
)
from bet_slip_watcher.main import (
    SlipProbe,
    _apply_probe_to_reading,
    _tag_wager_modal,
    analyze_frame,
    load_config,
    probe_slip,
)
from bet_slip_watcher.models import BetSlipReading
from bet_slip_watcher.ocr_engine import OcrEngine
from bet_slip_watcher.regions import RegionSet
from bet_slip_watcher.state import StateMachine, worth_outcome_notify

OUTCOME_EVENTS = frozenset({"bet_placing", "bet_placed"})


def evaluate_notify_events(image_path: Path, config_path: Path) -> list[str]:
    """
    Return outcome event names that would be sent (same gating as the live watcher).
    Empty list means no Telegram for bet_placing / bet_placed.
    """
    cfg = load_config(config_path)
    regions = RegionSet.from_config(cfg)
    hsv_cfg = cfg["place_bet_green_hsv"]
    ocr_cfg = cfg.get("ocr", {})
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    direct_slip = (
        "crop-slip" in image_path.name.lower()
        or max(frame.shape[:2]) < 900
    )
    if direct_slip:
        slip = frame
        panel = slip_panel_score(slip)
        settled_history = (
            bet_settled_history_heuristic(slip, hsv_cfg, panel=panel)
            or fullpage_in_slip_crop_heuristic(slip, hsv_cfg, panel=panel)
            or my_bets_list_in_slip_heuristic(slip, hsv_cfg, panel=panel)
            or sports_markets_page_in_slip(slip, panel=panel, hsv_cfg=hsv_cfg)
        )
        green = place_bet_green_ratio(slip, hsv_cfg)
        btn_green = place_bet_button_metrics(slip, hsv_cfg)
        placed_banner = (
            False
            if settled_history
            else bet_placed_modal_detected(
                slip, hsv_cfg, panel=panel, settled_history=settled_history
            )
        )
        probe = SlipProbe(
            panel=panel,
            green=green,
            btn_green=btn_green,
            placed_banner=placed_banner,
            wager_ui=False,
            settled_history=settled_history,
            slip_crop=slip,
        )
    else:
        probe = probe_slip(frame, regions, hsv_cfg)
        slip = probe.slip_crop

    promo = non_wager_popup_in_slip(slip, hsv_cfg, panel=probe.panel)
    placed_vis = bet_placed_modal_detected(
        slip, hsv_cfg, panel=probe.panel, settled_history=probe.settled_history
    )
    pre = place_bet_modal_idle(
        slip, hsv_cfg, panel=probe.panel, settled_history=probe.settled_history
    )
    placing_vis = bet_placing_heuristic(
        slip,
        hsv_cfg,
        probe.panel,
        pre,
        placed_vis,
        None,
        settled_history=probe.settled_history,
    )
    ocr = OcrEngine(
        languages=ocr_cfg.get("languages", ["eng"]),
        gpu=bool(ocr_cfg.get("gpu", False)),
        scale=float(ocr_cfg.get("scale", 2.0)),
    )
    submit = (
        not probe.settled_history
        and not promo
        and (
            probe.placed_banner
            or (
                not pre
                and probe.panel >= 0.45
                and (
                    probe.btn_green.visible
                    or probe.green.visible
                    or bet_placed_green_header(slip, hsv_cfg)
                )
            )
        )
    )
    if direct_slip and not probe.settled_history and not promo and not pre:
        if placing_vis or probe.placed_banner or placed_vis:
            submit = True

    if direct_slip:
        reading = BetSlipReading(slip_crop_score=probe.panel)
        reading = _apply_probe_to_reading(probe, reading, hsv_cfg)
        reading = _tag_wager_modal(reading, probe, hsv_cfg)
    else:
        _, reading = analyze_frame(
            frame,
            regions,
            ocr,
            hsv_cfg,
            ocr_cfg,
            run_ocr=submit,
            probe=probe,
        )

    markets_page = sports_markets_page_in_slip(
        slip, panel=probe.panel, hsv_cfg=hsv_cfg
    )
    if probe.settled_history or markets_page:
        reading = replace(
            reading,
            place_bet_visible=False,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    elif pre:
        reading = replace(
            reading,
            place_bet_visible=True,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    elif promo:
        reading = replace(
            reading,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    elif not submit and not probe.placed_banner and not placing_vis:
        reading = replace(
            reading,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    elif placing_vis and submit:
        reading = replace(
            reading,
            bet_placing_active=True,
            place_bet_visible=False,
            bet_placed_confirmed=False,
        )
    elif probe.placed_banner or placed_vis:
        reading = replace(
            reading,
            bet_placed_confirmed=True,
            place_bet_visible=False,
            bet_placing_active=False,
        )

    if not worth_outcome_notify(reading):
        return []
    sm = StateMachine(placed_confirm_frames=1)
    events = sm.observe(True, reading, reading.search_query, None)
    return [name for name, _ in events if name in OUTCOME_EVENTS]
