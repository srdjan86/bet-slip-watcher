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

from bet_slip_watcher.audit_frames import save_audit_frame
from dataclasses import dataclass, replace

from bet_slip_watcher.detect import (
    GreenButtonMetrics,
    action_bar_motion,
    action_bar_motion_sample,
    action_bar_white_on_green_ratio,
    bet_placed_banner_heuristic,
    bet_placing_heuristic,
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
    ocr_lines_show_place_bet,
    parse_market_row,
    parse_search_query,
    parse_slip_text,
)
from bet_slip_watcher.regions import RegionSet
from bet_slip_watcher.state import StateMachine, wager_key
from bet_slip_watcher.stream import (
    ImageFrameSource,
    OpenCvFrameSource,
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
        self._last_log_at = 0.0

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
        now = time.monotonic()
        if now - self._last_log_at >= 2.0:
            logger.info("%s", message)
            self._last_log_at = now

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
):
    if args.image:
        inner = ImageFrameSource(Path(args.image))
    elif args.stream or stream_url:
        source = args.stream or stream_url
        if "youtube" in source or "youtu.be" in source:
            cookie_path = streamlink_opts.cookie_path()
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
            elif cookie_path:
                logger.info("Opening YouTube via yt-dlp → ffmpeg (cookies file)…")
            else:
                logger.info("Opening YouTube via yt-dlp → ffmpeg…")
            inner = YtdlpPipeFrameSource(
                source,
                cookie_path,
                stream_from=stream_from,
                rewind_before_live=rewind_before_live,
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
        "bet_placed": "Bet placed on stream",
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
    placed_banner = bet_placed_banner_heuristic(slip_crop)
    wager_ui = wager_outcome_ui_active(panel, green, btn_green, placed_banner)
    return SlipProbe(
        panel=panel,
        green=green,
        btn_green=btn_green,
        placed_banner=placed_banner,
        wager_ui=wager_ui,
        slip_crop=slip_crop,
    )


def _slip_visible(
    probe: SlipProbe,
    slip: BetSlipReading,
) -> bool:
    return probe.panel > 0.25 and (
        slip.bet_placed_confirmed
        or slip.bet_placing_active
        or (probe.green.visible and slip.place_bet_visible)
        or (slip.stake_rsd and slip.stake_rsd >= 100 and probe.panel > 0.45)
        or probe.wager_ui
    )


def _apply_probe_to_reading(
    probe: SlipProbe,
    reading: BetSlipReading,
    hsv_cfg: dict,
) -> BetSlipReading:
    out = replace(reading, slip_crop_score=probe.panel)
    if probe.placed_banner:
        out.bet_placed_confirmed = True
        out.bet_placing_active = False
        out.place_bet_visible = False
    elif not out.bet_placed_confirmed:
        # Ignore stale "Place Bet" OCR while the spinner is showing.
        out.bet_placing_active = bet_placing_heuristic(
            probe.slip_crop,
            hsv_cfg,
            probe.panel,
            False,
            out.bet_placed_confirmed,
            out.stake_rsd,
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
    status_lines = ocr.read_lines(slip_crop[: int(sh * status_frac), :])
    slip_status = parse_slip_text(status_lines, place_bet_visible=False)

    search_lines: list[str] = []
    market_lines: list[str] = []
    if slip_status.bet_placed_confirmed or probe.placed_banner:
        slip_lines = list(status_lines)
        if not slip_status.bet_ref:
            slip_lines.extend(
                ocr.read_lines(slip_crop[int(sh * 0.22) : int(sh * 0.58), :])
            )
    else:
        market_lines = ocr.read_lines(regions.market_row.crop(frame))
        search_lines = ocr.read_lines(regions.search_bar.crop(frame))
        body_crop = slip_crop[: int(sh * 0.58), :]
        action_crop = slip_crop[int(sh * 0.45) :, :]
        slip_lines = ocr.read_lines(body_crop)
        slip_lines.extend(
            ocr.read_lines(action_crop, psm=7, scale=action_scale)
        )

    search_query = parse_search_query(search_lines)
    market_pick = parse_market_row(market_lines)
    slip = parse_slip_text(slip_lines, place_bet_visible=False)
    if slip.bet_placed_confirmed or probe.placed_banner:
        slip.place_bet_visible = False
        slip.bet_placed_confirmed = True
    elif green.visible and not slip.bet_placed_confirmed:
        slip.place_bet_visible = bool(
            slip.place_bet_visible
            or any(
                "place" in ln.lower() and "bet" in ln.lower() for ln in slip_lines
            )
        )

    slip.slip_crop_score = probe.panel
    slip.bet_placing_active = bet_placing_heuristic(
        slip_crop,
        hsv_cfg,
        probe.panel,
        slip.place_bet_visible,
        slip.bet_placed_confirmed,
        slip.stake_rsd,
    )
    if slip.place_bet_visible:
        slip.bet_placing_active = False

    if not slip.event_name:
        for line in market_lines:
            if " vs " in line.lower() or " ws " in line.lower():
                slip.event_name = line.replace(" ws ", " vs ").strip()
                break

    reading = merge_reading(search_query, market_pick, slip)
    return reading


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
    machine = StateMachine(
        slip_stable_frames=int(state_cfg.get("slip_stable_frames", 3)),
        placing_stable_frames=int(state_cfg.get("placing_stable_frames", 1)),
        placed_confirm_frames=int(state_cfg.get("placed_confirm_frames", 1)),
        placed_slip_gone_frames=int(state_cfg.get("placed_slip_gone_frames", 4)),
        likely_slip_gone_frames=int(state_cfg.get("likely_slip_gone_frames", 2)),
        notify_slip_ready=bool(state_cfg.get("notify_slip_ready", False)),
        notify_bet_placing=bool(state_cfg.get("notify_bet_placing", True)),
    )

    streamlink_opts = StreamlinkOptions.from_env()
    source = build_frame_source(args, stream_url, streamlink_opts)
    debug_dir = Path(args.debug_save_crops) if args.debug_save_crops else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = Path(args.audit_dir).expanduser() if getattr(args, "audit_dir", None) else None
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving audit frames to %s", audit_dir)

    logger.info("Watching… Ctrl+C to stop")
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
    awaiting_place_click = False
    action_bar_wog_peak = 0.0
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
            if wog >= place_bet_cta_min_wog:
                awaiting_place_click = True
                action_bar_wog_peak = wog
            else:
                awaiting_place_click = False
                latched_reading = None
                wager_ui_absent_streak = 0
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

        submit_started = probe.placed_banner or (
            probe.panel >= 0.45
            and (probe.btn_green.visible or probe.green.visible)
            and not awaiting_place_click
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

        ocr_start = latched_reading is None and submit_started and (
            probe.placed_banner
            or (submit_started and not prev_submit_started)
        )

        if ocr_once_per_wager:
            if ocr_start:
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
            _status_line(
                "frame=%s slip=%s placing=%s placed=%s sent=%s search=%s pick=%s stake=%s ref=%s"
                % (
                    idx,
                    slip_visible,
                    reading.bet_placing_active,
                    reading.bet_placed_confirmed,
                    machine.notifications_sent,
                    reading.search_query,
                    market_raw,
                    reading.stake_rsd,
                    reading.bet_ref,
                )
            )

        if debug_dir and idx < 5:
            cv2.imwrite(str(debug_dir / f"search_{idx}.png"), regions.search_bar.crop(frame))
            cv2.imwrite(str(debug_dir / f"market_{idx}.png"), regions.market_row.crop(frame))
            cv2.imwrite(str(debug_dir / f"slip_{idx}.png"), regions.bet_slip.crop(frame))

        if audit_dir:
            save_audit_frame(audit_dir, frame, idx, reading, slip_visible)
            if idx == 0 and getattr(args, "rewind", None):
                logger.info("First audit screenshot saved (rewind playback started)")

        for event_name, snap in events:
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
            logger.info(
                "Notify %s (total %s) key=%s",
                event_name,
                machine.notifications_sent,
                wager_key(snap)[:80],
            )
            await notifier.send(title, body)

    if args.verbose:
        _end_status_line()
    if frames_processed:
        logger.info("Finished processing %s frame(s)", frames_processed)


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
    args = parser.parse_args()
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
