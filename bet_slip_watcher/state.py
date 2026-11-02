from __future__ import annotations

import re
from dataclasses import dataclass, field

from bet_slip_watcher.models import BetSlipReading, WatcherPhase


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def wager_key(reading: BetSlipReading) -> str:
    """
    Stable wager id for dedupe. Omits event title (OCR noise) — uses pick + odds + stake.
    """
    sel = _norm(reading.selection_title)
    sel = re.sub(r"\s+\d\.\d{2}\s*$", "", sel)
    line = ""
    odds = reading.odds or (reading.market_pick.odds if reading.market_pick else "")
    if reading.market_pick and reading.market_pick.line:
        side = reading.market_pick.side or "?"
        line = f"{side}({reading.market_pick.line})"
    else:
        m = re.search(r"([+-]?\d+\.?\d*)", sel or "")
        if m and odds:
            line = f"?({m.group(1)})"
    stake = ""
    if reading.stake_rsd is not None and reading.stake_rsd >= 100:
        stake = str(int(round(reading.stake_rsd)))
    return "|".join([sel, line, str(odds), stake])


def notify_keys(reading: BetSlipReading) -> set[str]:
    keys = {f"wager:{wager_key(reading)}"}
    if reading.bet_ref:
        keys.add(f"ref:{reading.bet_ref.upper()}")
    return keys


def _normalize_spread_line(line: str | None) -> str:
    if not line:
        return ""
    m = re.search(r"([+-]?\d+\.?\d*)", str(line).replace(" ", ""))
    if not m:
        return _norm(line)
    v = float(m.group(1))
    if abs(v - round(v)) < 1e-6:
        iv = int(round(v))
        return f"{iv:+d}" if iv != 0 else "0"
    return f"{v:+g}"


def _strip_ocr_junk(s: str) -> str:
    s = _norm(s)
    return re.sub(r"^[^a-z0-9(]+", "", s)


def _clean_wager_text(s: str | None) -> str:
    if not s:
        return ""
    s = _strip_ocr_junk(s)
    s = re.sub(r"\d\.\d{2}\b", " ", s)
    s = re.sub(r"[+-]?\d+\.?\d*", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def selection_slug(reading: BetSlipReading) -> str:
    """Alphanumeric selection identity (spread/odds/OCR junk removed)."""
    slug = re.sub(r"[^a-z0-9]", "", _clean_wager_text(reading.selection_title))
    return slug[:32]


def wager_tokens(reading: BetSlipReading) -> set[str]:
    """Distinct name fragments for fuzzy same-match detection."""
    out: set[str] = set()
    for text in (
        reading.selection_title,
        reading.event_name,
        reading.search_query,
    ):
        cleaned = _clean_wager_text(text)
        for word in re.findall(r"[a-z0-9]{3,}", cleaned.replace(" ", " ")):
            if word.isdigit():
                continue
            if len(word) >= 3:
                out.add(word[:24])
        slug = re.sub(r"[^a-z0-9]", "", cleaned)
        if len(slug) >= 6:
            out.add(slug[:32])
    sel_slug = selection_slug(reading)
    if sel_slug:
        out.add(sel_slug)
    return out


def tokens_related(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return True
    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
        return True
    return len(a) >= 3 and len(b) >= 3 and (a in b or b in a)


def wager_tokens_overlap(a: BetSlipReading, b: BetSlipReading) -> bool:
    ta, tb = wager_tokens(a), wager_tokens(b)
    if ta & tb:
        return True
    for t in ta:
        for u in tb:
            if tokens_related(t, u):
                return True
    return False


def match_team_token(reading: BetSlipReading) -> str:
    """Short team hint from selection (legacy / logging)."""
    slug = selection_slug(reading)
    if slug:
        return slug[:20]
    for text in (reading.selection_title, reading.event_name):
        cleaned = _clean_wager_text(text)
        m = re.match(r"^([a-z0-9]{3,20})", re.sub(r"[^a-z0-9]", "", cleaned))
        if m:
            return m.group(1)
    return ""


def pick_fingerprint(reading: BetSlipReading) -> str:
    """Stable id: selection slug + spread line + odds."""
    mp = reading.market_pick
    odds = reading.odds or (mp.odds if mp else "") or ""
    line = _spread_line_from_reading(reading)
    slug = selection_slug(reading)
    if slug and line and odds:
        return f"{slug}|{line}|{odds}"
    team = match_team_token(reading)
    if team and line and odds:
        return f"{team}|{line}|{odds}"
    if mp and mp.line and odds:
        side = mp.side or "?"
        return f"{side}|{_normalize_spread_line(mp.line)}|{odds}"
    if line and odds:
        return f"?|{line}|{odds}"
    return wager_key(reading)


def _spread_line_from_reading(reading: BetSlipReading) -> str:
    if reading.market_pick and reading.market_pick.line:
        return _normalize_spread_line(reading.market_pick.line)
    m = re.search(r"([+-]?\d+\.?\d*)", _norm(reading.selection_title or ""))
    return _normalize_spread_line(m.group(1)) if m else ""


def same_wager(a: BetSlipReading, b: BetSlipReading) -> bool:
    if a.bet_ref and b.bet_ref and a.bet_ref.upper() == b.bet_ref.upper():
        return True
    fa, fb = pick_fingerprint(a), pick_fingerprint(b)
    if fa and fa == fb:
        return True
    oa = a.odds or (a.market_pick.odds if a.market_pick else "")
    ob = b.odds or (b.market_pick.odds if b.market_pick else "")
    la, lb = _spread_line_from_reading(a), _spread_line_from_reading(b)
    if la and lb and la == lb and oa and ob and oa == ob:
        if wager_tokens_overlap(a, b):
            return True
        sa, sb = selection_slug(a), selection_slug(b)
        if sa and sb and tokens_related(sa, sb):
            return True
    return wager_key(a) == wager_key(b)


_OUTCOME_EVENTS = ("bet_placing", "bet_placed", "bet_likely_placed")


def worth_outcome_notify(reading: BetSlipReading) -> bool:
    """Skip Telegram when OCR did not capture a real wager (search UI, false banner)."""
    if reading.my_bets_history_page:
        return False
    if reading.bet_placing_active or reading.bet_placed_confirmed:
        if not reading.wager_modal_compact and not reading.bet_ref:
            return False
    if reading.bet_ref:
        return True
    odds = reading.odds or (
        reading.market_pick.odds if reading.market_pick else ""
    )
    if reading.market_pick and reading.market_pick.line and odds:
        return True
    if reading.market_pick and reading.market_pick.raw and odds:
        return True
    sel = _norm(reading.selection_title)
    if len(sel) > 6 and odds:
        return True
    if sel and reading.stake_rsd is not None and reading.stake_rsd >= 100:
        return True
    return False


@dataclass
class StateMachine:
    slip_stable_frames: int = 3
    placing_stable_frames: int = 1
    placed_confirm_frames: int = 1
    placed_slip_gone_frames: int = 4
    likely_slip_gone_frames: int = 2
    notify_slip_ready: bool = False
    notify_bet_placing: bool = True

    phase: WatcherPhase = WatcherPhase.IDLE
    stable_count: int = 0
    placing_count: int = 0
    placed_confirm_count: int = 0
    gone_count: int = 0
    ui_clear_count: int = 0
    last_ready_reading: BetSlipReading | None = None
    last_wager_reading: BetSlipReading | None = None
    notified_keys: set[str] = field(default_factory=set)
    active_wager_key: str | None = None
    placing_telegram_sent: bool = False
    last_outcome_fingerprint: str | None = None
    last_outcome_reading: BetSlipReading | None = None
    last_outcome_tokens: set[str] = field(default_factory=set)

    def _event_keys(self, event: str, reading: BetSlipReading) -> set[str]:
        return {f"{event}:{k}" for k in notify_keys(reading)}

    def _already_notified_event(self, event: str, reading: BetSlipReading) -> bool:
        return bool(self._event_keys(event, reading) & self.notified_keys)

    def _notify_once(
        self,
        event: str,
        reading: BetSlipReading,
        out: list[tuple[str, BetSlipReading]],
    ) -> bool:
        """Send at most one Telegram per event type per wager / bet ref."""
        keys = self._event_keys(event, reading)
        if keys & self.notified_keys:
            return False
        out.append((event, reading))
        self.notified_keys.update(keys)
        self.active_wager_key = wager_key(reading)
        return True

    def _outcome_already_notified(self, event: str, reading: BetSlipReading) -> bool:
        """bet_placing and bet_placed for the same pick count as one wager."""
        fp = pick_fingerprint(reading)
        if fp and f"outcome:{fp}" in self.notified_keys:
            return True
        if self._already_notified_event(event, reading):
            return True
        if fp and self.last_outcome_fingerprint and fp == self.last_outcome_fingerprint:
            return True
        if self.last_outcome_reading and same_wager(reading, self.last_outcome_reading):
            return True
        if self.last_outcome_tokens and wager_tokens(reading):
            oa = reading.odds or (
                reading.market_pick.odds if reading.market_pick else ""
            )
            la = _spread_line_from_reading(reading)
            if self.last_outcome_reading and la and oa:
                lr = self.last_outcome_reading
                lo = lr.odds or (lr.market_pick.odds if lr.market_pick else "")
                ll = _spread_line_from_reading(lr)
                if lo == oa and ll == la:
                    for t in wager_tokens(reading):
                        for prev in self.last_outcome_tokens:
                            if tokens_related(t, prev):
                                return True
        for other in _OUTCOME_EVENTS:
            if other == event:
                continue
            if self._already_notified_event(other, reading):
                return True
        return False

    def _notify_outcome_once(
        self,
        event: str,
        raw: BetSlipReading,
        snap: BetSlipReading,
        out: list[tuple[str, BetSlipReading]],
    ) -> bool:
        if not worth_outcome_notify(raw):
            return False
        if self._outcome_already_notified(event, snap):
            return False
        if not self._notify_once(event, snap, out):
            return False
        fp = pick_fingerprint(snap)
        if fp:
            self.notified_keys.add(f"outcome:{fp}")
        self.last_outcome_fingerprint = fp or None
        self.last_outcome_reading = snap
        self.last_outcome_tokens = wager_tokens(snap)
        if event == "bet_placing":
            self.placing_telegram_sent = True
        return True

    def clear_outcome_dedupe(self) -> None:
        """Allow outcome Telegram again (new bet after slip UI cleared or new Place Bet)."""
        self.notified_keys = {
            k for k in self.notified_keys if not k.startswith("outcome:")
        }
        self.last_outcome_fingerprint = None
        self.last_outcome_reading = None
        self.last_outcome_tokens = set()
        self.placing_telegram_sent = False

    def _merge_wager_context(self, reading: BetSlipReading) -> BetSlipReading:
        base = self.last_wager_reading or self.last_ready_reading
        if base is None:
            return reading
        if not worth_outcome_notify(reading) and (
            reading.bet_placing_active or reading.bet_placed_confirmed
        ):
            return reading
        # New wager on the same stream — do not inherit the previous slip OCR.
        if reading.bet_placing_active or reading.place_bet_visible:
            if _norm(reading.selection_title) and wager_key(reading) != wager_key(base):
                return reading
        if wager_key(reading) != wager_key(base) and (
            reading.bet_placing_active or reading.bet_placed_confirmed
        ):
            return reading
        for attr in (
            "search_query",
            "market_pick",
            "selection_title",
            "event_name",
            "odds",
            "stake_rsd",
            "to_return_rsd",
            "market_type",
        ):
            if getattr(reading, attr) in (None, "", 0.0) and getattr(base, attr):
                setattr(reading, attr, getattr(base, attr))
        reading.bet_ref = reading.bet_ref or base.bet_ref
        return reading

    def _ui_cleared(self, reading: BetSlipReading, slip_visible: bool) -> None:
        busy = (
            slip_visible
            or reading.bet_placed_confirmed
            or reading.bet_placing_active
        )
        if busy:
            self.ui_clear_count = 0
            return
        self.ui_clear_count += 1
        if self.ui_clear_count >= self.placed_slip_gone_frames:
            self.active_wager_key = None
            self.last_wager_reading = None
            self.last_ready_reading = None
            self.placed_confirm_count = 0
            self.placing_count = 0
            self.clear_outcome_dedupe()
            self.gone_count = 0
            self.ui_clear_count = 0
            self.phase = WatcherPhase.IDLE

    @property
    def notifications_sent(self) -> int:
        return len({k for k in self.notified_keys if ":" in k and k.split(":", 1)[0] in {
            "bet_placing",
            "bet_placed",
            "slip_ready",
            "bet_likely_placed",
        }})

    def begin_wager_cycle(self) -> None:
        """Reset OCR/confirm counters for spinner → placed on the same wager."""
        self.placing_count = 0
        self.placed_confirm_count = 0

    def bet_placed_already_sent(self, reading: BetSlipReading) -> bool:
        return self._outcome_already_notified("bet_placed", reading)

    def observe(
        self,
        slip_visible: bool,
        reading: BetSlipReading,
        search_query: str | None,
        market_raw: str | None,
    ) -> list[tuple[str, BetSlipReading]]:
        out: list[tuple[str, BetSlipReading]] = []

        if search_query:
            reading.search_query = reading.search_query or search_query
        if market_raw and reading.market_pick is None:
            pass

        snap = self._merge_wager_context(reading)

        if reading.bet_placed_confirmed:
            self.placing_count = 0
            self.placed_confirm_count += 1
            snap.bet_placed_confirmed = True
            self.last_wager_reading = snap
            if self.placed_confirm_count >= self.placed_confirm_frames:
                self._notify_outcome_once("bet_placed", reading, snap, out)
                self.phase = WatcherPhase.IDLE
                self.last_ready_reading = None
                self.placed_confirm_count = 0
            return out

        self.placed_confirm_count = 0

        if reading.bet_placing_active:
            self.placing_count += 1
            self.last_wager_reading = snap
            if self.placing_count >= self.placing_stable_frames:
                if (
                    self.notify_bet_placing
                    and not self.placing_telegram_sent
                    and not reading.place_bet_visible
                ):
                    self._notify_outcome_once("bet_placing", reading, snap, out)
                self.phase = WatcherPhase.AWAITING_CONFIRM
            return out

        self.placing_count = 0
        self._ui_cleared(reading, slip_visible)

        ready = (
            slip_visible
            and reading.place_bet_visible
            and not reading.bet_placed_confirmed
            and reading.stake_rsd is not None
            and reading.stake_rsd > 0
        )

        if ready:
            self.stable_count += 1
            self.gone_count = 0
            self.last_ready_reading = snap
            self.last_wager_reading = snap
            if (
                self.notify_slip_ready
                and self.phase == WatcherPhase.IDLE
                and self.stable_count >= self.slip_stable_frames
            ):
                self._notify_once("slip_ready", snap, out)
                self.phase = WatcherPhase.SLIP_OPEN
        elif self.phase in (WatcherPhase.SLIP_OPEN, WatcherPhase.AWAITING_CONFIRM):
            if not slip_visible:
                self.gone_count += 1
                self.stable_count = 0
                fallback = self.last_wager_reading or self.last_ready_reading
                gone_need = (
                    self.likely_slip_gone_frames
                    if self.phase == WatcherPhase.AWAITING_CONFIRM or self.last_wager_reading
                    else self.placed_slip_gone_frames
                )
                if fallback is not None and self.gone_count >= gone_need:
                    self._notify_outcome_once("bet_likely_placed", fallback, fallback, out)
                    self.phase = WatcherPhase.IDLE
                    self.last_ready_reading = None
                    self.gone_count = 0
            else:
                self.gone_count = 0
        else:
            self.stable_count = 0
            self.gone_count = 0

        return out
