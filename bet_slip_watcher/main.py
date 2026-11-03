from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv

from bet_slip_watcher.audit_frames import (
    encode_crop_slip_png,
    save_audit_frame,
    save_review_capture,
    save_timeline_snapshot,
)
from dataclasses import dataclass, replace

from bet_slip_watcher.detect import (
    GreenButtonMetrics,
    action_bar_motion,
    action_bar_motion_sample,
    action_bar_white_on_green_ratio,
    bet_placed_banner_heuristic,
    bet_placed_modal_detected,
    bet_placed_green_header,
    bet_placing_heuristic,
    bet_settled_history_heuristic,
    compact_wager_modal_heuristic,
    non_wager_popup_in_slip,
    place_bet_modal_idle,
    wager_modal_light_body_ratio,
    fullpage_in_slip_crop_heuristic,
    my_bets_list_in_slip_heuristic,
    place_bet_cta_visible,
    place_bet_green_ratio,
    place_bet_button_metrics,
    slip_panel_score,
    wager_outcome_ui_active,
)
from bet_slip_watcher.models import BetSlipReading
from bet_slip_watcher.notify import Notifier, NotifierConfig
from bet_slip_watcher.ocr_engine import OcrEngine
from bet_slip_watcher.parse import (
    merge_reading,
    ocr_lines_indicate_my_bets_settled,
    ocr_lines_show_place_bet,
    parse_market_row,
    parse_search_query,
    parse_slip_text,
)
from bet_slip_watcher.regions import RegionSet
from bet_slip_watcher.state import StateMachine, wager_key
from bet_slip_watcher.stream import (
    ImageFrameSource,
    LatestOnlyFrameSource,
    OpenCvFrameSource,
    YtdlpAuth,
    StreamlinkOptions,
    ThrottledFrameSource,
    YtdlpPipeFrameSource,
    parse_rewind_duration,
    resolve_stream_url,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("bet_slip_watcher")

_CLEAR_LINE = "\033[2K\r"


class _VerboseStatus:
    """One live status line; falls back to throttled logs if the terminal ignores \\r."""

    def __init__(self) -> None:
        plain = os.environ.get("BET_SLIP_VERBOSE_PLAIN", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._stream = None if plain else (sys.stdout if sys.stdout.isatty() else None)

    def update(self, message: str) -> None:
        if self._stream is not None:
            try:
                width = shutil.get_terminal_size(fallback=(100, 24)).columns
            except OSError:
                width = 100
            pad = max(0, width - len(message) - 1)
            self._stream.write(_CLEAR_LINE + message + " " * pad)
            self._stream.flush()
            return
        # Non-TTY (piped logs): skip per-frame status spam; startup line is enough.

    def finish(self) -> None:
        if self._stream is not None:
            self._stream.write("\n")
            self._stream.flush()


_verbose_status = _VerboseStatus()


def _status_line(message: str) -> None:
    _verbose_status.update(message)


def _end_status_line() -> None:
    _verbose_status.finish()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_frame_source(
    args: argparse.Namespace,
    stream_url: str,
    streamlink_opts: StreamlinkOptions,
    *,
    live_latest_frame: bool = True,
):
    if args.image:
        inner = ImageFrameSource(Path(args.image))
    elif args.stream or stream_url:
        source = args.stream or stream_url
        if "youtube" in source or "youtu.be" in source:
            ytdlp_auth = YtdlpAuth.from_env()
            stream_from = (getattr(args, "stream_from", None) or "").strip() or None
            rewind_before_live = None
            rewind_label = (getattr(args, "rewind", None) or "").strip()
            if rewind_label:
                rewind_before_live = parse_rewind_duration(rewind_label)
                logger.info(
                    "Opening YouTube rewound %s (%ss before live edge)…",
                    rewind_label,
                    rewind_before_live,
                )
            elif stream_from:
                logger.info(
                    "Opening YouTube from %s (DVR rewind via yt-dlp)…",
                    stream_from,
                )
            elif ytdlp_auth.cookies_from_browser:
                logger.info(
                    "Opening YouTube via yt-dlp → ffmpeg (browser profile %s)…",
                    ytdlp_auth.cookies_from_browser,
                )
            elif ytdlp_auth.cookie_file:
                logger.info("Opening YouTube via yt-dlp → ffmpeg (cookies file)…")
            else:
                logger.info("Opening YouTube via yt-dlp → ffmpeg…")
            inner = YtdlpPipeFrameSource(
                source,
                auth=ytdlp_auth if ytdlp_auth.configured() else None,
                stream_from=stream_from,
                rewind_before_live=rewind_before_live,
            )
            if (
                live_latest_frame
                and not inner._dvr_rewind
            ):
                inner = LatestOnlyFrameSource(inner)
                logger.info(
                    "Live YouTube: analyze newest frame only (avoid decode backlog lag)"
                )
        else:
            path = Path(source).expanduser()
            is_local_video = (
                path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi", ".ts"}
                and path.is_file()
            )
            if is_local_video:
                fps = float(args.fps)
                logger.info(
                    "Analyzing local video %s (full length, ~%.1f samples/sec)",
                    path,
                    fps,
                )
                return OpenCvFrameSource(source, sample_fps=fps)
            inner = OpenCvFrameSource(source)
    else:
        raise SystemExit("Set STREAM_URL / --stream or --image for local debug")

    fps = float(args.fps)
    return ThrottledFrameSource(inner, fps)


def format_event(title: str, reading: BetSlipReading) -> tuple[str, str]:
    headline = {
        "slip_ready": "Bet slip ready (friend about to place?)",
        "bet_placing": "Bet placing on stream",
        "bet_placed": "",
        "bet_likely_placed": "Bet likely placed on stream",
    }.get(title, title)
    body = "\n".join(reading.summary_lines())
    if not body:
        body = "(OCR empty — tune regions in config.yaml)"
    return headline, body


@dataclass
class SlipProbe:
    panel: float
    green: GreenButtonMetrics
    btn_green: GreenButtonMetrics
    placed_banner: bool
    wager_ui: bool
    settled_history: bool
    slip_crop: object  # np.ndarray


def action_bar_says_place_bet(
    slip_crop,
    ocr: OcrEngine,
    *,
    scale: float,
) -> bool:
    from bet_slip_watcher.detect import action_bar_crop

    lines = ocr.read_lines(action_bar_crop(slip_crop), psm=7, scale=scale)
    return ocr_lines_show_place_bet(lines)


def _finalize_reading(
    probe: SlipProbe,
    reading: BetSlipReading,
    hsv_cfg: dict,
    *,
    awaiting_place_click: bool,
) -> BetSlipReading:
    if probe.settled_history:
        return replace(
            reading,
            bet_placing_active=False,
            bet_placed_confirmed=False,
            place_bet_visible=False,
            my_bets_history_page=True,
        )
    if awaiting_place_click:
        return replace(
            reading,
            place_bet_visible=True,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    if reading.place_bet_visible:
        return replace(reading, bet_placing_active=False)
    return _apply_probe_to_reading(probe, reading, hsv_cfg)


def probe_slip(frame, regions: RegionSet, hsv_cfg: dict) -> SlipProbe:
    slip_crop = regions.bet_slip.crop(frame)
    green = place_bet_green_ratio(slip_crop, hsv_cfg)
    btn_green = place_bet_button_metrics(slip_crop, hsv_cfg)
    panel = slip_panel_score(slip_crop)
    settled_history = (
        bet_settled_history_heuristic(slip_crop, hsv_cfg, panel=panel)
        or fullpage_in_slip_crop_heuristic(slip_crop, hsv_cfg, panel=panel)
        or my_bets_list_in_slip_heuristic(slip_crop, hsv_cfg, panel=panel)
    )
    placed_banner = (
        False
        if settled_history
        else bet_placed_modal_detected(
            slip_crop, hsv_cfg, panel=panel, settled_history=settled_history
        )
    )
    wager_ui = (
        False
        if settled_history
        else wager_outcome_ui_active(panel, green, btn_green, placed_banner)
    )
    return SlipProbe(
        panel=panel,
        green=green,
        btn_green=btn_green,
        placed_banner=placed_banner,
        wager_ui=wager_ui,
        settled_history=settled_history,
        slip_crop=slip_crop,
    )


def _slip_visible(
    probe: SlipProbe,
    slip: BetSlipReading,
) -> bool:
    if probe.settled_history:
        return False
    return probe.panel > 0.25 and (
        slip.bet_placed_confirmed
        or slip.bet_placing_active
        or (probe.green.visible and slip.place_bet_visible)
        or (slip.stake_rsd and slip.stake_rsd >= 100 and probe.panel > 0.45)
        or probe.wager_ui
    )


def _tag_wager_modal(
    reading: BetSlipReading,
    probe: SlipProbe,
    hsv_cfg: dict,
) -> BetSlipReading:
    return replace(
        reading,
        wager_modal_compact=compact_wager_modal_heuristic(
            probe.slip_crop,
            hsv_cfg,
            panel=probe.panel,
            settled_history=probe.settled_history,
        ),
    )


def _apply_probe_to_reading(
    probe: SlipProbe,
    reading: BetSlipReading,
    hsv_cfg: dict,
) -> BetSlipReading:
    out = replace(reading, slip_crop_score=probe.panel)
    out = _tag_wager_modal(out, probe, hsv_cfg)
    if probe.settled_history:
        return replace(
            out,
            bet_placing_active=False,
            bet_placed_confirmed=False,
            place_bet_visible=False,
            my_bets_history_page=True,
        )
    pre_bet = place_bet_modal_idle(
        probe.slip_crop,
        hsv_cfg,
        panel=probe.panel,
        settled_history=probe.settled_history,
    )
    if pre_bet:
        return replace(
            out,
            place_bet_visible=True,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    if probe.placed_banner:
        out.bet_placed_confirmed = True
        out.bet_placing_active = False
        out.place_bet_visible = False
    elif not out.bet_placed_confirmed:
        out.bet_placing_active = bet_placing_heuristic(
            probe.slip_crop,
            hsv_cfg,
            probe.panel,
            out.place_bet_visible,
            out.bet_placed_confirmed,
            out.stake_rsd,
            settled_history=probe.settled_history,
        )
        if out.bet_placing_active:
            out.place_bet_visible = False
    return out


def ocr_wager_frame(
    frame,
    regions: RegionSet,
    ocr: OcrEngine,
    hsv_cfg: dict,
    probe: SlipProbe,
    ocr_cfg: dict | None = None,
) -> BetSlipReading:
    """Full Tesseract pass — call once per wager on rising edge into post-submit UI."""
    opts = ocr_cfg or {}
    action_scale = float(opts.get("action_bar_scale", 3.5))
    slip_crop = probe.slip_crop
    green = probe.green
    sh, _sw = slip_crop.shape[:2]
    status_frac = float(opts.get("placed_status_fraction", 0.40))
    header_scale = float(opts.get("placed_header_scale", 4.0))
    status_lines = ocr.read_lines(slip_crop[: int(sh * status_frac), :])
    if probe.placed_banner or bet_placed_green_header(slip_crop, hsv_cfg):
        header_lines = ocr.read_lines(
            slip_crop[: max(12, int(sh * 0.24)), :],
            psm=6,
            scale=header_scale,
        )
        status_lines = header_lines + status_lines
    slip_status = parse_slip_text(status_lines, place_bet_visible=False)

    search_lines: list[str] = []
    market_lines: list[str] = []
    search_lines = ocr.read_lines(regions.search_bar.crop(frame))
    market_lines = ocr.read_lines(regions.market_row.crop(frame))
    footer_scale = float(opts.get("placed_footer_scale", 4.0))
    if slip_status.bet_placed_confirmed or probe.placed_banner:
        slip_lines = list(status_lines)
        slip_lines.extend(
            ocr.read_lines(slip_crop[int(sh * 0.18) : int(sh * 0.58), :])
        )
        slip_lines.extend(
            ocr.read_lines(
                slip_crop[int(sh * 0.28) : int(sh * 0.48), :],
                psm=6,
                scale=footer_scale,
            )
        )
        slip_lines.extend(
            ocr.read_lines(
                slip_crop[int(sh * 0.58) :, :],
                psm=6,
                scale=footer_scale,
            )
        )
    else:
        body_crop = slip_crop[: int(sh * 0.58), :]
        action_crop = slip_crop[int(sh * 0.45) :, :]
        slip_lines = ocr.read_lines(body_crop)
        slip_lines.extend(
            ocr.read_lines(action_crop, psm=7, scale=action_scale)
        )

    search_query = parse_search_query(search_lines)
    market_pick = parse_market_row(market_lines)
    slip = parse_slip_text(slip_lines, place_bet_visible=False)
    pre_bet = place_bet_modal_idle(
        slip_crop,
        hsv_cfg,
        panel=probe.panel,
        settled_history=probe.settled_history,
    )
    if pre_bet:
        slip = replace(
            slip,
            place_bet_visible=True,
            bet_placing_active=False,
            bet_placed_confirmed=False,
        )
    elif slip.bet_placed_confirmed or probe.placed_banner:
        slip.place_bet_visible = False
        slip.bet_placed_confirmed = probe.placed_banner or (
            slip.bet_placed_confirmed
            and compact_wager_modal_heuristic(
                slip_crop,
                hsv_cfg,
                panel=probe.panel,
                settled_history=probe.settled_history,
            )
        )
    elif green.visible and not slip.bet_placed_confirmed:
        slip.place_bet_visible = bool(
            slip.place_bet_visible
            or any(
                "place" in ln.lower() and "bet" in ln.lower() for ln in slip_lines
            )
        )

    slip.slip_crop_score = probe.panel
    if pre_bet or slip.place_bet_visible:
        slip.bet_placing_active = False
        slip.bet_placed_confirmed = False
    elif slip.bet_placed_confirmed or probe.placed_banner:
        slip.bet_placing_active = False
    else:
        slip.bet_placing_active = bet_placing_heuristic(
            slip_crop,
            hsv_cfg,
            probe.panel,
            slip.place_bet_visible,
            slip.bet_placed_confirmed,
            slip.stake_rsd,
            settled_history=probe.settled_history,
        )
        if slip.place_bet_visible:
            slip.bet_placing_active = False

    if not slip.event_name:
        for line in market_lines:
            if " vs " in line.lower() or " ws " in line.lower():
                slip.event_name = line.replace(" ws ", " vs ").strip()
                break

    if (
        not pre_bet
        and not probe.placed_banner
        and not bet_placed_modal_detected(
            slip_crop, hsv_cfg, panel=probe.panel, settled_history=probe.settled_history
        )
        and not slip.bet_ref
    ):
        slip.bet_placed_confirmed = False

    reading = merge_reading(search_query, market_pick, slip)
    history_page = probe.settled_history or ocr_lines_indicate_my_bets_settled(
        reading.ocr_lines
    )
    if history_page:
        reading.my_bets_history_page = True
        reading.bet_placing_active = False
        reading.bet_placed_confirmed = False
        reading.place_bet_visible = False
    return _tag_wager_modal(reading, probe, hsv_cfg)


def reading_without_ocr(probe: SlipProbe) -> BetSlipReading:
    slip = BetSlipReading(slip_crop_score=probe.panel)
    if probe.placed_banner:
        slip.bet_placed_confirmed = True
    return slip


def analyze_frame(
    frame,
    regions: RegionSet,
    ocr: OcrEngine,
    hsv_cfg: dict,
    ocr_cfg: dict | None = None,
    *,
    run_ocr: bool = False,
    latched: BetSlipReading | None = None,
    probe: SlipProbe | None = None,
    awaiting_place_click: bool = False,
) -> tuple[bool, BetSlipReading]:
    opts = ocr_cfg or {}
    ocr_market_when_idle = bool(opts.get("ocr_market_when_idle", False))
    p = probe or probe_slip(frame, regions, hsv_cfg)

    if run_ocr:
        reading = ocr_wager_frame(frame, regions, ocr, hsv_cfg, p, ocr_cfg)
        reading = _finalize_reading(
            p, reading, hsv_cfg, awaiting_place_click=False
        )
        return _slip_visible(p, reading), reading

    if latched is not None:
        reading = _finalize_reading(
            p, latched, hsv_cfg, awaiting_place_click=awaiting_place_click
        )
        return _slip_visible(p, reading), reading

    reading = reading_without_ocr(p)
    reading = _finalize_reading(
        p, reading, hsv_cfg, awaiting_place_click=awaiting_place_click
    )
    if ocr_market_when_idle and not p.wager_ui:
        market_lines = ocr.read_lines(regions.market_row.crop(frame))
        market_pick = parse_market_row(market_lines)
        reading = merge_reading(None, market_pick, reading)
    return _slip_visible(p, reading), reading


async def run_watcher(args: argparse.Namespace) -> None:
    load_dotenv()
    if not getattr(args, "stream_from", None):
        env_from = os.getenv("STREAM_FROM", "").strip()
        if env_from:
            args.stream_from = env_from
    if not getattr(args, "rewind", None):
        env_rewind = os.getenv("REWIND", "").strip()
        if env_rewind:
            args.rewind = env_rewind
    if args.rewind and args.stream_from:
        raise SystemExit("Use either --rewind or --stream-from, not both")
    if not getattr(args, "audit_dir", None):
        env_audit = os.getenv("AUDIT_DIR", "").strip()
        if env_audit:
            args.audit_dir = env_audit
    if args.rewind and not args.audit_dir:
        args.audit_dir = "./audit"
        logger.info(
            "Rewind mode: saving every analyzed frame to ./audit (override with --audit-dir)"
        )
    cfg = load_config(Path(args.config))
    regions = RegionSet.from_config(cfg)
    hsv_cfg = cfg["place_bet_green_hsv"]
    state_cfg = cfg.get("state", {})
    ocr_cfg = cfg.get("ocr", {})

    stream_url = args.stream or os.getenv("STREAM_URL", "")
    if not args.image and not stream_url:
        raise SystemExit("STREAM_URL missing — copy .env.example to .env")

    ocr = OcrEngine(
        languages=ocr_cfg.get("languages", ["eng"]),
        gpu=bool(ocr_cfg.get("gpu", False)),
        scale=float(ocr_cfg.get("scale", 2.0)),
    )
    notifier = Notifier(
        NotifierConfig(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            ntfy_topic=os.getenv("NTFY_TOPIC", ""),
        )
    )
    telegram_attach_crop_slip = bool(
        state_cfg.get("telegram_attach_crop_slip", True)
    )
    slip_export = {
        "tighten": bool(state_cfg.get("slip_crop_tighten", True)),
        "pad": int(state_cfg.get("slip_crop_tighten_pad", 12)),
        "min_col_density": float(state_cfg.get("slip_crop_min_col_density", 0.08)),
    }
    machine = StateMachine(
        slip_stable_frames=int(state_cfg.get("slip_stable_frames", 3)),
        placing_stable_frames=int(state_cfg.get("placing_stable_frames", 1)),
        placed_confirm_frames=int(state_cfg.get("placed_confirm_frames", 1)),
        placed_slip_gone_frames=int(state_cfg.get("placed_slip_gone_frames", 4)),
        likely_slip_gone_frames=int(state_cfg.get("likely_slip_gone_frames", 2)),
        notify_slip_ready=bool(state_cfg.get("notify_slip_ready", False)),
        notify_bet_placing=bool(state_cfg.get("notify_bet_placing", True)),
        notify_bet_likely_placed=bool(state_cfg.get("notify_bet_likely_placed", False)),
    )

    streamlink_opts = StreamlinkOptions.from_env()
    live_latest = bool(cfg.get("live_latest_frame", True))
    source = build_frame_source(
        args, stream_url, streamlink_opts, live_latest_frame=live_latest
    )
    debug_dir = Path(args.debug_save_crops) if args.debug_save_crops else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = Path(args.audit_dir).expanduser() if getattr(args, "audit_dir", None) else None
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving audit frames to %s", audit_dir)

    review_dir: Path | None = None
    if getattr(args, "no_review_screenshots", False):
        pass
    elif getattr(args, "review_dir", None) is not None:
        review_dir = Path(args.review_dir).expanduser()
    else:
        env_review = os.getenv("REVIEW_DIR", "").strip()
        if env_review:
            review_dir = Path(env_review).expanduser()
        else:
            review_cfg = state_cfg.get("review_screenshots_dir", "./review")
            if review_cfg not in (False, None, ""):
                review_dir = Path(str(review_cfg)).expanduser()
    if review_dir:
        review_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving review screenshots on each notify to %s",
            review_dir,
        )

    timeline_dir: Path | None = None
    timeline_interval = float(state_cfg.get("timeline_interval_seconds", 10))
    if getattr(args, "no_timeline_screenshots", False):
        pass
    elif getattr(args, "timeline_interval", None) is not None:
        timeline_interval = float(args.timeline_interval)
    if not getattr(args, "no_timeline_screenshots", False):
        if getattr(args, "timeline_dir", None) is not None:
            timeline_dir = Path(args.timeline_dir).expanduser()
        else:
            env_timeline = os.getenv("TIMELINE_DIR", "").strip()
            if env_timeline:
                timeline_dir = Path(env_timeline).expanduser()
            else:
                timeline_cfg = state_cfg.get("timeline_screenshots_dir", "./timeline")
                if timeline_cfg not in (False, None, ""):
                    timeline_dir = Path(str(timeline_cfg)).expanduser()
    if timeline_dir:
        timeline_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving timeline screenshot every %.0fs to %s",
            timeline_interval,
            timeline_dir,
        )

    watch_started = time.monotonic()
    last_timeline_save = watch_started
    max_frames = int(args.max_frames or 0)
    frames_processed = 0
    latched_reading: BetSlipReading | None = None
    wager_ui_absent_streak = 0
    wager_ui_gone_frames = int(state_cfg.get("placed_slip_gone_frames", 4))
    ocr_once_per_wager = bool(ocr_cfg.get("once_per_wager", True))
    action_bar_scale = float(ocr_cfg.get("action_bar_scale", 3.5))
    click_motion_min = float(ocr_cfg.get("place_bet_click_motion", 8.0))
    spinner_drop_ratio = float(ocr_cfg.get("action_bar_spinner_drop_ratio", 0.45))
    place_bet_cta_min_wog = float(ocr_cfg.get("place_bet_cta_min_wog", 0.022))
    prev_btn_visible = False
    prev_action_motion_sample = None
    prev_submit_started = False
    prev_placed_banner = False
    awaiting_place_click = False
    action_bar_wog_peak = 0.0
    startup_warmup_frames = int(state_cfg.get("startup_warmup_frames", 8))
    logger.info("Watching… Ctrl+C to stop")
    for idx, frame in enumerate(source.frames()):
        if max_frames and idx >= max_frames:
            break
        frames_processed = idx + 1
        probe = probe_slip(frame, regions, hsv_cfg)
        action_motion_sample = action_bar_motion_sample(probe.slip_crop)

        btn_visible = probe.btn_green.visible and probe.panel >= 0.45
        btn_rising = btn_visible and not prev_btn_visible
        wog = action_bar_white_on_green_ratio(probe.slip_crop, hsv_cfg)
        if not btn_visible and not probe.placed_banner:
            awaiting_place_click = False
            action_bar_wog_peak = 0.0
            prev_action_motion_sample = None
        elif btn_rising and not probe.placed_banner:
            latched_reading = None
            machine.clear_outcome_dedupe()
            machine.begin_wager_cycle()
            wager_ui_absent_streak = 0
            if place_bet_modal_idle(
                probe.slip_crop,
                hsv_cfg,
                panel=probe.panel,
                settled_history=probe.settled_history,
            ) or wog >= place_bet_cta_min_wog or (
                probe.panel <= 0.93
                and wager_modal_light_body_ratio(probe.slip_crop) >= 0.10
                and probe.btn_green.visible
            ):
                awaiting_place_click = True
                action_bar_wog_peak = wog
            else:
                awaiting_place_click = False
        elif awaiting_place_click and btn_visible:
            action_bar_wog_peak = max(action_bar_wog_peak, wog)
            if not place_bet_cta_visible(
                probe.slip_crop,
                hsv_cfg,
                action_bar_wog_peak,
                spinner_drop_ratio=spinner_drop_ratio,
            ):
                awaiting_place_click = False
        elif awaiting_place_click and prev_action_motion_sample is not None:
            motion = action_bar_motion(
                prev_action_motion_sample, action_motion_sample
            )
            if motion >= click_motion_min and btn_visible:
                awaiting_place_click = False
        if awaiting_place_click and not btn_visible and probe.panel >= 0.45:
            if not probe.placed_banner:
                awaiting_place_click = False
        if probe.placed_banner:
            awaiting_place_click = False
            action_bar_wog_peak = 0.0
        if btn_visible or probe.placed_banner:
            prev_action_motion_sample = action_motion_sample

        submit_started = (
            not probe.settled_history
            and not non_wager_popup_in_slip(
                probe.slip_crop,
                hsv_cfg,
                panel=probe.panel,
            )
            and (
                probe.placed_banner
                or (
                    not place_bet_modal_idle(
                        probe.slip_crop,
                        hsv_cfg,
                        panel=probe.panel,
                        settled_history=probe.settled_history,
                    )
                    and probe.panel >= 0.45
                    and (
                        probe.btn_green.visible
                        or probe.green.visible
                        or bet_placed_green_header(probe.slip_crop, hsv_cfg)
                    )
                    and not awaiting_place_click
                )
            )
        )

        session_visible = submit_started or probe.placed_banner
        if session_visible:
            wager_ui_absent_streak = 0
        else:
            wager_ui_absent_streak += 1
            if latched_reading is not None:
                if latched_reading.bet_placed_confirmed:
                    if wager_ui_absent_streak >= wager_ui_gone_frames:
                        latched_reading = None
                elif wager_ui_absent_streak >= wager_ui_gone_frames * 12:
                    latched_reading = None

        in_wager_session = latched_reading is not None and (
            session_visible or wager_ui_absent_streak < wager_ui_gone_frames * 12
        )

        placed_edge = probe.placed_banner and not prev_placed_banner
        submit_edge = submit_started and not prev_submit_started
        if placed_edge:
            latched_reading = None
            machine.begin_wager_cycle()
        elif submit_edge:
            latched_reading = None
            machine.begin_wager_cycle()

        if probe.placed_banner:
            preview = (
                _apply_probe_to_reading(probe, latched_reading, hsv_cfg)
                if latched_reading is not None
                else reading_without_ocr(probe)
            )
            if not machine.bet_placed_already_sent(preview):
                latched_reading = None
                machine.begin_wager_cycle()

        ocr_start = submit_started and (
            placed_edge
            or submit_edge
            or (latched_reading is None and probe.placed_banner)
        )

        if ocr_once_per_wager:
            if ocr_start:
                if not (placed_edge or submit_edge):
                    machine.begin_wager_cycle()
                slip_visible, reading = analyze_frame(
                    frame,
                    regions,
                    ocr,
                    hsv_cfg,
                    ocr_cfg,
                    run_ocr=True,
                    probe=probe,
                )
                latched_reading = reading
            elif in_wager_session and latched_reading is not None:
                slip_visible, reading = analyze_frame(
                    frame,
                    regions,
                    ocr,
                    hsv_cfg,
                    ocr_cfg,
                    run_ocr=False,
                    latched=latched_reading,
                    probe=probe,
                    awaiting_place_click=awaiting_place_click,
                )
                if reading.bet_placed_confirmed and not latched_reading.bet_placed_confirmed:
                    if not wager_key(reading).replace("|", "").strip():
                        slip_visible, reading = analyze_frame(
                            frame,
                            regions,
                            ocr,
                            hsv_cfg,
                            ocr_cfg,
                            run_ocr=True,
                            probe=probe,
                        )
                    latched_reading = reading
            else:
                slip_visible, reading = analyze_frame(
                    frame,
                    regions,
                    ocr,
                    hsv_cfg,
                    ocr_cfg,
                    run_ocr=False,
                    probe=probe,
                    awaiting_place_click=awaiting_place_click,
                )
        else:
            slip_visible, reading = analyze_frame(
                frame,
                regions,
                ocr,
                hsv_cfg,
                ocr_cfg,
                run_ocr=submit_started and not awaiting_place_click,
                probe=probe,
                awaiting_place_click=awaiting_place_click,
            )

        prev_btn_visible = btn_visible
        prev_submit_started = submit_started
        prev_placed_banner = probe.placed_banner

        pre_bet_idle = place_bet_modal_idle(
            probe.slip_crop,
            hsv_cfg,
            panel=probe.panel,
            settled_history=probe.settled_history,
        )
        if pre_bet_idle or awaiting_place_click:
            reading = replace(
                reading,
                place_bet_visible=True,
                bet_placing_active=False,
                bet_placed_confirmed=False,
            )
        elif non_wager_popup_in_slip(
            probe.slip_crop,
            hsv_cfg,
            panel=probe.panel,
        ):
            reading = replace(
                reading,
                place_bet_visible=False,
                bet_placing_active=False,
                bet_placed_confirmed=False,
            )
        elif probe.placed_banner or bet_placed_modal_detected(
            probe.slip_crop,
            hsv_cfg,
            panel=probe.panel,
            settled_history=probe.settled_history,
        ):
            reading = replace(
                reading,
                bet_placed_confirmed=True,
                place_bet_visible=False,
                bet_placing_active=False,
            )

        still_in_wager = latched_reading is not None and (
            session_visible or wager_ui_absent_streak < wager_ui_gone_frames * 12
        )
        if ocr_once_per_wager and still_in_wager:
            slip_visible = True

        market_raw = reading.market_pick.raw if reading.market_pick else None
        events = machine.observe(
            slip_visible,
            reading,
            reading.search_query,
            market_raw,
        )

        if args.verbose:
            _status_line("Running…")

        if debug_dir and idx < 5:
            cv2.imwrite(str(debug_dir / f"search_{idx}.png"), regions.search_bar.crop(frame))
            cv2.imwrite(str(debug_dir / f"market_{idx}.png"), regions.market_row.crop(frame))
            cv2.imwrite(str(debug_dir / f"slip_{idx}.png"), regions.bet_slip.crop(frame))

        if audit_dir:
            save_audit_frame(audit_dir, frame, idx, reading, slip_visible)
            if idx == 0 and getattr(args, "rewind", None):
                logger.info("First audit screenshot saved (rewind playback started)")

        if timeline_dir:
            now = time.monotonic()
            if now - last_timeline_save >= timeline_interval:
                elapsed = now - watch_started
                shot = save_timeline_snapshot(
                    timeline_dir,
                    frame,
                    idx,
                    elapsed_s=elapsed,
                    placed_banner=probe.placed_banner,
                    bet_placed=reading.bet_placed_confirmed,
                    bet_placing=reading.bet_placing_active,
                    awaiting_click=awaiting_place_click,
                    has_latch=latched_reading is not None,
                    btn_visible=btn_visible,
                    submit_started=submit_started,
                )
                last_timeline_save = now
                logger.info(
                    "Timeline screenshot (t=%.0fs frame=%s): %s",
                    elapsed,
                    idx,
                    shot.name,
                )

        for event_name, snap in events:
            if idx < startup_warmup_frames and event_name in (
                "bet_placed",
                "bet_placing",
            ):
                if args.verbose:
                    logger.info(
                        "Skipping startup %s (frame %s < warmup %s)",
                        event_name,
                        idx,
                        startup_warmup_frames,
                    )
                continue
            if args.verbose:
                _end_status_line()
            if audit_dir:
                save_audit_frame(
                    audit_dir,
                    frame,
                    idx,
                    snap,
                    slip_visible,
                    prefix=f"NOTIFY-{event_name}",
                )
            title, body = format_event(event_name, snap)
            if review_dir:
                shot = save_review_capture(
                    review_dir,
                    frame,
                    idx,
                    snap,
                    event_name=event_name,
                    regions=regions,
                    tighten_slip=slip_export["tighten"],
                    tighten_pad=slip_export["pad"],
                    tighten_min_col=slip_export["min_col_density"],
                )
                logger.info(
                    "Review screenshot (notify %s): %s",
                    event_name,
                    shot.name,
                )
            logger.info(
                "Notify %s (total %s) key=%s",
                event_name,
                machine.notifications_sent,
                wager_key(snap)[:80],
            )
            photo: bytes | None = None
            if telegram_attach_crop_slip:
                photo = encode_crop_slip_png(frame, regions, **slip_export)
            await notifier.send(title, body, photo_png=photo)

    if args.verbose:
        _end_status_line()
    if frames_processed:
        logger.info("Finished processing %s frame(s)", frames_processed)


def check_image_path(image_path: Path, config_path: Path) -> int:
    """Run detection + notify gating on a screenshot; print verdict (exit 0 = no notify)."""
    from bet_slip_watcher.state import StateMachine, worth_outcome_notify

    cfg = load_config(config_path)
    regions = RegionSet.from_config(cfg)
    hsv_cfg = cfg["place_bet_green_hsv"]
    ocr_cfg = cfg.get("ocr", {})
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: cannot read {image_path}")
        return 2
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
    if direct_slip:
        reading = BetSlipReading(slip_crop_score=probe.panel)
        reading = _apply_probe_to_reading(probe, reading, hsv_cfg)
        reading = _tag_wager_modal(reading, probe, hsv_cfg)
        slip_visible = probe.panel > 0.25 and (
            reading.bet_placed_confirmed or reading.bet_placing_active or placing_vis
        )
    else:
        _, reading = analyze_frame(
            frame, regions, ocr, hsv_cfg, ocr_cfg, run_ocr=submit, probe=probe
        )
        slip_visible = True
    if pre:
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
    elif probe.placed_banner or placed_vis:
        reading = replace(
            reading,
            bet_placed_confirmed=True,
            place_bet_visible=False,
            bet_placing_active=False,
        )
    events: list[tuple[str, object]] = []
    if worth_outcome_notify(reading):
        sm = StateMachine(placed_confirm_frames=1)
        events = sm.observe(True, reading, reading.search_query, None)
    print(f"image: {image_path.name}")
    print(
        f"  settled={probe.settled_history} promo={promo} pre_bet={pre} "
        f"placed_vis={placed_vis} placing_vis={placing_vis} probe_banner={probe.placed_banner}"
    )
    print(
        f"  reading: placed={reading.bet_placed_confirmed} placing={reading.bet_placing_active} "
        f"stake={reading.stake_rsd} ref={reading.bet_ref!r}"
    )
    if events:
        print(f"  NOTIFY: {[e[0] for e in events]}")
        return 1
    print("  NOTIFY: (none) — not treated as a bet")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch bet365 stream for bet slip → notify with search/pick/stake context",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--stream", help="YouTube or direct stream URL (overrides .env)")
    parser.add_argument(
        "--image",
        help="Local screenshot loop for tuning (no YouTube)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Sample FPS")
    parser.add_argument(
        "--rewind",
        metavar="DURATION",
        help=(
            "Start playback N before live edge (e.g. 4h15m, 4h, 1h, 4:15:00). "
            "Saves audit screenshots to ./audit by default."
        ),
    )
    parser.add_argument(
        "--stream-from",
        metavar="TIME",
        help="YouTube DVR scrub-bar time H:MM:SS (advanced; prefer --rewind 4h)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run forever)",
    )
    parser.add_argument(
        "--debug-save-crops",
        metavar="DIR",
        help="Save first frames' OCR crops for tuning regions",
    )
    parser.add_argument(
        "--audit-dir",
        metavar="DIR",
        help="Save every analyzed frame with detection labels in the filename (for manual review)",
    )
    parser.add_argument(
        "--review-dir",
        metavar="DIR",
        help="Folder for bet-placed / OCR-empty screenshots (default: config review_screenshots_dir)",
    )
    parser.add_argument(
        "--no-review-screenshots",
        action="store_true",
        help="Do not save review screenshots on bet placed or empty OCR",
    )
    parser.add_argument(
        "--timeline-dir",
        metavar="DIR",
        help="Folder for periodic timeline screenshots (default: config timeline_screenshots_dir)",
    )
    parser.add_argument(
        "--timeline-interval",
        type=float,
        metavar="SEC",
        help="Seconds between timeline screenshots (default: config timeline_interval_seconds)",
    )
    parser.add_argument(
        "--no-timeline-screenshots",
        action="store_true",
        help="Do not save periodic timeline screenshots",
    )
    parser.add_argument(
        "--check-image",
        metavar="PATH",
        help="Print whether this screenshot would trigger bet_placing/bet_placed (no Telegram)",
    )
    args = parser.parse_args()
    if args.check_image:
        raise SystemExit(check_image_path(Path(args.check_image).expanduser(), Path(args.config)))
    if args.fps is None:
        cfg = load_config(Path(args.config))
        args.fps = float(cfg.get("sample_fps", 2))

    try:
        asyncio.run(run_watcher(args))
    except KeyboardInterrupt:
        if args.verbose:
            _end_status_line()
        logger.info("Stopped")


if __name__ == "__main__":
    main()
