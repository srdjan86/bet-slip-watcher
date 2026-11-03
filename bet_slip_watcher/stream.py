from __future__ import annotations

import logging
import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YtdlpAuth:
    """
    YouTube auth for yt-dlp.

    Prefer ``YTDLP_COOKIES_FROM_BROWSER`` on Mac (dedicated Chrome profile) so
    cookies refresh from the browser DB — no manual cookies.txt export.
    """

    cookie_file: Path | None = None
    cookies_from_browser: str | None = None

    @classmethod
    def from_env(cls) -> YtdlpAuth:
        browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
        if browser:
            return cls(cookies_from_browser=browser)
        raw = os.getenv("STREAMLINK_HTTP_COOKIE_FILE", "").strip()
        if not raw:
            return cls()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"STREAMLINK_HTTP_COOKIE_FILE not found: {path} "
                "(or set YTDLP_COOKIES_FROM_BROWSER — see README)"
            )
        return cls(cookie_file=path)

    def configured(self) -> bool:
        return bool(self.cookies_from_browser or self.cookie_file)

    def ytdlp_args(self) -> list[str]:
        if self.cookies_from_browser:
            return ["--cookies-from-browser", self.cookies_from_browser]
        if self.cookie_file:
            return ["--cookies", str(self.cookie_file)]
        return []


@dataclass(frozen=True)
class StreamlinkOptions:
    """Auth for private/restricted streams via streamlink's --http-cookies-file (v8+)."""

    http_cookie_file: str = ""

    @classmethod
    def from_env(cls) -> StreamlinkOptions:
        return cls(
            http_cookie_file=os.getenv("STREAMLINK_HTTP_COOKIE_FILE", "").strip(),
        )

    def cookie_path(self) -> Path | None:
        auth = YtdlpAuth.from_env()
        if auth.cookies_from_browser:
            return None
        if not self.http_cookie_file:
            return auth.cookie_file
        path = Path(self.http_cookie_file).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"STREAMLINK_HTTP_COOKIE_FILE not found: {path} "
                "(export YouTube-only cookies.txt — see README)"
            )
        return path

    def streamlink_prefix_args(self) -> list[str]:
        path = self.cookie_path()
        if path is None:
            return []
        return ["--http-cookies-file", str(path)]


def parse_stream_timestamp(value: str) -> int:
    """Parse H:MM:SS, M:SS, or seconds into total seconds."""
    raw = value.strip()
    if not raw:
        raise ValueError("empty timestamp")
    if raw.isdigit():
        return int(raw)
    parts = raw.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(float(seconds))
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(float(seconds))
    raise ValueError(f"invalid timestamp: {value!r}")


_REWIND_DURATION = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$",
    re.IGNORECASE,
)
_REWIND_COMPOUND = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)",
    re.IGNORECASE,
)


def _rewind_unit_seconds(num: float, unit: str) -> int:
    u = unit.lower()
    if u.startswith("h"):
        return int(num * 3600)
    if u.startswith("m"):
        return int(num * 60)
    return int(num)


def parse_rewind_duration(value: str) -> int:
    """Parse ``4h15m``, ``4h``, ``30m``, or ``4:15:00`` / ``H:MM:SS`` into seconds."""
    raw = value.strip()
    if not raw:
        raise ValueError("empty rewind duration")
    compound = _REWIND_COMPOUND.findall(raw)
    if len(compound) >= 2 or (len(compound) == 1 and not _REWIND_DURATION.match(raw.replace(" ", ""))):
        total = sum(_rewind_unit_seconds(float(n), u) for n, u in compound)
        if total > 0:
            return total
    compact = raw.replace(" ", "")
    match = _REWIND_DURATION.match(compact)
    if match:
        return _rewind_unit_seconds(float(match.group(1)), match.group(2) or "h")
    return parse_stream_timestamp(raw)


def seek_before_live(
    youtube_url: str,
    auth: YtdlpAuth | None,
    before_live_seconds: int,
) -> tuple[int, str]:
    """Map “N seconds before live edge” to YouTube DVR scrub-bar seconds."""
    release = fetch_youtube_release_timestamp(youtube_url, auth)
    if release is None:
        raise RuntimeError(
            "Could not read when the live stream started (yt-dlp release_timestamp). "
            "Check cookies and that the stream is live."
        )
    elapsed = max(0, int(time.time()) - release)
    seek = elapsed - before_live_seconds
    if seek < 0:
        raise RuntimeError(
            f"Stream has been live ~{elapsed // 3600}h{(elapsed % 3600) // 60}m — "
            f"cannot rewind {before_live_seconds // 3600}h"
            f"{(before_live_seconds % 3600) // 60}m before now"
        )
    label = (
        f"{before_live_seconds // 3600}h{(before_live_seconds % 3600) // 60:02d}m "
        f"before live (scrub {seek // 3600}h{(seek % 3600) // 60:02d}m from start)"
    )
    return seek, label


def format_rewind_label(before_live_seconds: int) -> str:
    h = before_live_seconds // 3600
    m = (before_live_seconds % 3600) // 60
    return f"{h}h{m:02d}m before live edge"


def dvr_section_spec(
    *,
    rewind_before_live: int | None = None,
    stream_from: str | None = None,
    youtube_url: str,
    auth: YtdlpAuth | None,
) -> tuple[str, str]:
    """
    Build yt-dlp --download-sections value and a log label.

    ``--rewind`` uses *-Ns-inf (seconds before live edge). YouTube's DVR window
    is only a few hours — absolute scrub time from stream start breaks on long
    broadcasts.
    """
    if rewind_before_live is not None:
        if rewind_before_live <= 0:
            raise ValueError("rewind duration must be positive")
        return (
            f"*-{rewind_before_live}-inf",
            format_rewind_label(rewind_before_live),
        )
    if not stream_from:
        raise ValueError("stream_from required when rewind_before_live is None")
    raw = stream_from.strip()
    if raw.startswith("#-"):
        before_live = parse_stream_timestamp(raw[2:])
        return f"*-{before_live}-inf", format_rewind_label(before_live)
    seek = parse_stream_timestamp(raw)
    return f"*{seek}-inf", f"scrub bar {raw} ({seek}s from start)"


def youtube_url_with_start(url: str, start_seconds: int) -> str:
    if start_seconds <= 0:
        return url
    if "t=" in url or "start=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={start_seconds}"


def _pick_playback_url(stdout: str) -> str:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("empty stream URL")
    for line in lines:
        if ".m3u8" in line:
            return line
    return lines[0]


def fetch_youtube_release_timestamp(
    youtube_url: str,
    auth: YtdlpAuth | None,
) -> int | None:
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--skip-download",
        "--print",
        "release_timestamp",
    ]
    if auth:
        cmd.extend(auth.ytdlp_args())
    cmd.append(youtube_url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return int(text) if text.isdigit() else None


def resolve_stream_from_seconds(
    stream_from: str,
    youtube_url: str,
    auth: YtdlpAuth | None,
) -> tuple[int, str]:
    """
    Return (seek_seconds, log_label).

    ``4:13:00`` = offset on the YouTube scrub bar from stream start.
    ``#-1:30:00`` = that long before the current live edge (needs release_timestamp).
    """
    raw = stream_from.strip()
    if raw.startswith("#-"):
        before_live = parse_stream_timestamp(raw[2:])
        release = fetch_youtube_release_timestamp(youtube_url, auth)
        if release is None:
            raise RuntimeError(
                "Could not read stream start time for #-relative rewind. "
                "Use scrub-bar time instead, e.g. --stream-from 4:13:00"
            )
        elapsed = max(0, int(time.time()) - release)
        seek = elapsed - before_live
        if seek < 0:
            raise RuntimeError(
                f"Stream has run ~{elapsed // 3600}h{(elapsed % 3600) // 60}m; "
                f"cannot rewind {before_live // 3600}h{(before_live % 3600) // 60}m before live"
            )
        return seek, f"{raw} ({seek}s on scrub bar)"
    if raw.startswith("#"):
        raise RuntimeError(
            "Use #-H:MM:SS for time before live edge, or H:MM:SS for scrub-bar time "
            "(e.g. --stream-from 4:13:00)"
        )
    seek = parse_stream_timestamp(raw)
    return seek, raw


def resolve_youtube_stream_url_ytdlp(
    youtube_url: str,
    auth: YtdlpAuth,
    *,
    live_from_start: bool = False,
) -> str:
    """Private/restricted YouTube — streamlink often returns LOGIN_REQUIRED; yt-dlp works."""
    cmd = [
        "yt-dlp",
        "-g",
        "--no-warnings",
    ]
    cmd.extend(auth.ytdlp_args())
    if live_from_start:
        cmd.append("--live-from-start")
    cmd.append(youtube_url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown error").strip()
        raise RuntimeError(f"yt-dlp failed: {err}")
    return _pick_playback_url(proc.stdout)


def resolve_stream_url(
    youtube_url: str,
    options: StreamlinkOptions | None = None,
) -> str:
    """Resolve a YouTube URL to a direct HLS/HTTP URL (yt-dlp if cookies, else streamlink)."""
    auth = YtdlpAuth.from_env()
    if auth.configured():
        return resolve_youtube_stream_url_ytdlp(youtube_url, auth)
    opts = options or StreamlinkOptions()
    cookie_path = opts.cookie_path()
    if cookie_path is not None:
        return resolve_youtube_stream_url_ytdlp(
            youtube_url, YtdlpAuth(cookie_file=cookie_path)
        )

    cmd = ["streamlink", "--stream-url", youtube_url, "best"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown error").strip()
        raise RuntimeError(f"streamlink failed: {err}")
    url = proc.stdout.strip()
    if not url:
        raise RuntimeError("streamlink returned empty stream URL")
    return url


class FrameSource:
    def frames(self) -> Generator[np.ndarray, None, None]:
        raise NotImplementedError


class OpenCvFrameSource(FrameSource):
    """Local files only — OpenCV cannot reliably open YouTube HLS URLs on macOS."""

    def __init__(self, url_or_path: str, *, sample_fps: float | None = None) -> None:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            raise RuntimeError(
                "HTTP(S) streams must use yt-dlp/ffmpeg pipe, not OpenCV VideoCapture"
            )
        self._source = url_or_path
        self._sample_fps = sample_fps
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._source}")
        return cap

    def frames(self) -> Generator[np.ndarray, None, None]:
        self._cap = self._open()
        try:
            step = 1
            if self._sample_fps:
                video_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
                if video_fps <= 0:
                    video_fps = 30.0
                step = max(1, round(video_fps / self._sample_fps))
                total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total > 0:
                    logger.info(
                        "Local video ~%.1f fps, %s frames, emitting every %s frame(s)",
                        video_fps,
                        total,
                        step,
                    )
            index = 0
            emitted = 0
            while True:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    break
                if index % step == 0:
                    emitted += 1
                    yield frame
                index += 1
            logger.info(
                "Local video ended (%s frames read, %s sampled)",
                index,
                emitted,
            )
        finally:
            if self._cap is not None:
                self._cap.release()


def _dimensions_from_frame_bytes(data: bytes) -> tuple[int, int]:
    pixels = len(data) // 3
    for height in (1080, 720, 1440, 900, 2160):
        if pixels % height:
            continue
        width = pixels // height
        if width * height * 3 == len(data):
            return width, height
    raise RuntimeError(f"Cannot infer frame size from {len(data)} bytes of video data")


class YtdlpPipeFrameSource(FrameSource):
    """
    Decode YouTube (incl. private) by piping yt-dlp into ffmpeg.
    Avoids passing expiring HLS URLs to OpenCV/ffprobe (403 without yt-dlp).
    """

    def __init__(
        self,
        youtube_url: str,
        auth: YtdlpAuth | None = None,
        stream_from: str | None = None,
        rewind_before_live: int | None = None,
    ) -> None:
        self._auth = auth
        self._youtube_url = youtube_url
        self._stream_from = (stream_from or "").strip() or None
        self._section_spec: str | None = None
        self._dvr_rewind = False
        label = ""
        if rewind_before_live is not None:
            release = fetch_youtube_release_timestamp(youtube_url, auth)
            if release is not None:
                elapsed = max(0, int(time.time()) - release)
                if rewind_before_live > elapsed:
                    raise RuntimeError(
                        f"Stream has been live ~{elapsed // 3600}h"
                        f"{(elapsed % 3600) // 60}m — cannot rewind that far back"
                    )
            self._section_spec, label = dvr_section_spec(
                rewind_before_live=rewind_before_live,
                stream_from=None,
                youtube_url=youtube_url,
                auth=auth,
            )
            self._dvr_rewind = True
        elif self._stream_from:
            self._section_spec, label = dvr_section_spec(
                rewind_before_live=None,
                stream_from=self._stream_from,
                youtube_url=youtube_url,
                auth=auth,
            )
            self._dvr_rewind = True
        if self._dvr_rewind:
            assert self._section_spec is not None
            logger.info("Rewind: %s", label)
            logger.info(
                "DVR: yt-dlp --download-sections %s",
                self._section_spec,
            )
            logger.info(
                "Seeking… first screenshot may take 1–3 minutes (logs every ~15s until then)"
            )
        self._width, self._height = 1920, 1080

    def _ytdlp_cmd(self) -> list[str]:
        cmd = [
            "yt-dlp",
            "--no-warnings",
            "--no-progress",
            "-f",
            "best",
            "-o",
            "-",
        ]
        if self._dvr_rewind:
            cmd.extend(
                [
                    "--download-sections",
                    self._section_spec,
                    "--hls-use-mpegts",
                ]
            )
        else:
            cmd.append("--quiet")
        if self._auth:
            cmd.extend(self._auth.ytdlp_args())
        cmd.append(self._youtube_url)
        return cmd

    def _start_pipeline(self) -> tuple[subprocess.Popen, subprocess.Popen]:
        ytdlp = subprocess.Popen(
            self._ytdlp_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-an",
                "-sn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=ytdlp.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert ytdlp.stdout is not None
        ytdlp.stdout.close()
        return ytdlp, ffmpeg

    def _read_ffmpeg_rawvideo(
        self,
        ffmpeg: subprocess.Popen,
        *,
        error_context: str,
        ytdlp: subprocess.Popen | None = None,
        dvr_seek: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        frame_bytes = self._width * self._height * 3
        assert ffmpeg.stdout is not None
        got_frame = False
        pending = bytearray()
        wait_start = time.monotonic()
        last_log = wait_start
        fd = ffmpeg.stdout.fileno()

        def _poll_process_errors() -> None:
            if ytdlp is not None and ytdlp.poll() is not None and ytdlp.returncode:
                err = (ytdlp.stderr.read() if ytdlp.stderr else b"").decode(
                    errors="replace"
                )
                for line in err.splitlines():
                    if line.strip().startswith("ERROR:"):
                        raise RuntimeError(
                            f"Cannot open YouTube stream: {line.strip()}"
                        )
                    if "Cannot match chapters" in line:
                        raise RuntimeError(
                            "DVR rewind failed at that timestamp — try a shorter "
                            "--rewind (YouTube may not keep that much DVR)."
                        )
                raise RuntimeError(
                    "Cannot open YouTube stream "
                    "(check cookies, stream live, yt-dlp/ffmpeg installed)"
                )
            if ffmpeg.poll() is not None and ffmpeg.returncode:
                err = (ffmpeg.stderr.read() if ffmpeg.stderr else b"").decode(
                    errors="replace"
                ).strip()
                detail = f": {err}" if err else ""
                raise RuntimeError(f"Cannot open YouTube stream ({error_context}){detail}")

        while True:
            while len(pending) < frame_bytes:
                if dvr_seek and not got_frame:
                    now = time.monotonic()
                    if now - last_log >= 15.0:
                        logger.info(
                            "Still seeking DVR (%.0fs elapsed)…",
                            now - wait_start,
                        )
                        last_log = now
                ready, _, _ = select.select([ffmpeg.stdout], [], [], 15.0)
                if not ready:
                    _poll_process_errors()
                    continue
                chunk = os.read(fd, min(262144, frame_bytes - len(pending)))
                if not chunk:
                    _poll_process_errors()
                    break
                pending.extend(chunk)

            if len(pending) >= frame_bytes:
                raw = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
            else:
                raw = bytes(pending)
                pending.clear()

            if len(raw) == frame_bytes:
                got_frame = True
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self._height, self._width, 3
                )
                yield frame.copy()
                continue
            if len(raw) > 1000 and len(raw) % 3 == 0:
                try:
                    w, h = _dimensions_from_frame_bytes(raw)
                    if w * h * 3 == len(raw):
                        self._width, self._height = w, h
                        frame_bytes = len(raw)
                        got_frame = True
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                        yield frame.copy()
                        continue
                except RuntimeError:
                    pass
            if not got_frame:
                _poll_process_errors()
            break

    def _frames_ytdlp_pipe(self) -> Generator[np.ndarray, None, None]:
        reconnect = not self._dvr_rewind
        while True:
            ytdlp, ffmpeg = self._start_pipeline()
            try:
                for frame in self._read_ffmpeg_rawvideo(
                    ffmpeg,
                    error_context="yt-dlp pipe",
                    ytdlp=ytdlp,
                    dvr_seek=self._dvr_rewind,
                ):
                    yield frame
                if not reconnect:
                    logger.info("DVR replay ended (live edge or stream stopped)")
                    break
                logger.warning("YouTube pipe ended — reconnecting to live…")
            finally:
                for proc in (ffmpeg, ytdlp):
                    proc.kill()
                    proc.wait(timeout=5)
            if not reconnect:
                break
            time.sleep(2.0)

    def frames(self) -> Generator[np.ndarray, None, None]:
        yield from self._frames_ytdlp_pipe()


class ImageFrameSource(FrameSource):
    def __init__(self, path: Path) -> None:
        self._path = path

    def frames(self) -> Generator[np.ndarray, None, None]:
        img = cv2.imread(str(self._path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {self._path}")
        while True:
            yield img.copy()
            time.sleep(1.0)


class LatestOnlyFrameSource(FrameSource):
    """
    Background decode; consumer always analyzes the newest frame.
    Prevents multi-minute lag when OCR/processing is slower than video fps.
    """

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner

    def frames(self) -> Generator[np.ndarray, None, None]:
        import threading

        condition = threading.Condition()
        latest: np.ndarray | None = None
        error: BaseException | None = None
        stopped = False

        def produce() -> None:
            nonlocal latest, error, stopped
            try:
                for frame in self._inner.frames():
                    with condition:
                        latest = frame
                        condition.notify_all()
            except BaseException as exc:
                with condition:
                    error = exc
                    condition.notify_all()
            finally:
                with condition:
                    stopped = True
                    condition.notify_all()

        threading.Thread(
            target=produce, daemon=True, name="latest-frame-decode"
        ).start()

        while True:
            with condition:
                while latest is None and error is None and not stopped:
                    condition.wait(timeout=1.0)
                if error is not None:
                    raise error
                if latest is None:
                    break
                frame = latest
                latest = None
            yield frame.copy()


class ThrottledFrameSource(FrameSource):
    """Emit at most ``target_fps`` frames per second (one inner frame per tick)."""

    def __init__(self, inner: FrameSource, target_fps: float) -> None:
        self._inner = inner
        self._interval = 1.0 / max(target_fps, 0.1)

    def frames(self) -> Generator[np.ndarray, None, None]:
        next_tick = time.monotonic()
        for frame in self._inner.frames():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + self._interval, time.monotonic())
            yield frame
