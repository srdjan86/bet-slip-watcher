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

    def _merge_wager_context(self, reading: BetSlipReading) -> BetSlipReading:
        base = self.last_wager_reading or self.last_ready_reading
        if base is None:
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
            self.placing_telegram_sent = False
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
        """New bet OCR session (spinner or placed) — allow placing alert again."""
        self.placing_telegram_sent = False
        self.placing_count = 0
        self.placed_confirm_count = 0

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

        if reading.bet_placing_active:
            self.placing_count += 1
            self.last_wager_reading = snap
            if self.placing_count >= self.placing_stable_frames:
                if (
                    self.notify_bet_placing
                    and not self.placing_telegram_sent
                    and not reading.place_bet_visible
                ):
                    if self._notify_once("bet_placing", snap, out):
                        self.placing_telegram_sent = True
                self.phase = WatcherPhase.AWAITING_CONFIRM
            return out

        self.placing_count = 0

        if reading.bet_placed_confirmed:
            self.placing_count = 0
            self.placed_confirm_count += 1
            snap.bet_placed_confirmed = True
            self.last_wager_reading = snap
            if self.placed_confirm_count >= self.placed_confirm_frames:
                if not self.placing_telegram_sent:
                    self._notify_once("bet_placed", snap, out)
                self.phase = WatcherPhase.IDLE
                self.last_ready_reading = None
                self.placed_confirm_count = 0
            return out

        self.placed_confirm_count = 0
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
                    self._notify_once("bet_likely_placed", fallback, out)
                    self.phase = WatcherPhase.IDLE
                    self.last_ready_reading = None
                    self.gone_count = 0
            else:
                self.gone_count = 0
        else:
            self.stable_count = 0
            self.gone_count = 0

        return out
