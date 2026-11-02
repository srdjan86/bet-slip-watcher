from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WatcherPhase(str, Enum):
    IDLE = "idle"
    SLIP_OPEN = "slip_open"
    AWAITING_CONFIRM = "awaiting_confirm"


@dataclass
class MarketPick:
    """What was clicked on the coupon row, e.g. side 1 at +16.5 @ 1.83."""

    side: Optional[str] = None  # "1" or "2"
    line: Optional[str] = None  # "+16.5"
    odds: Optional[str] = None  # "1.83"
    raw: str = ""


@dataclass
class BetSlipReading:
    search_query: Optional[str] = None
    market_pick: Optional[MarketPick] = None
    selection_title: Optional[str] = None  # "Earth Friends Tokyo Z +16.5"
    market_type: Optional[str] = None  # "Point Spread"
    event_name: Optional[str] = None
    odds: Optional[str] = None
    stake_rsd: Optional[float] = None
    to_return_rsd: Optional[float] = None
    place_bet_visible: bool = False
    bet_placed_confirmed: bool = False
    bet_placing_active: bool = False
    bet_ref: Optional[str] = None
    slip_crop_score: float = 0.0
    my_bets_history_page: bool = False
    wager_modal_compact: bool = False
    ocr_lines: list[str] = field(default_factory=list)

    def _pick_line_for_notify(self) -> str | None:
        mp = self.market_pick
        if mp is None:
            return None
        if mp.side and mp.line:
            return f"{mp.side} ({mp.line})"
        if mp.side:
            return mp.side
        if mp.raw:
            return re.sub(r"\s*\d\.\d{2}\s*$", "", mp.raw.strip())
        return None

    def _selection_for_notify(self) -> str | None:
        if not self.selection_title:
            return None
        return re.sub(r"\s+\d\.\d{2}\s*$", "", self.selection_title.strip())

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.search_query:
            lines.append(f"🔍 {self.search_query}")
        if self.event_name:
            lines.append(f"🏀 {self.event_name}")
        pick = self._pick_line_for_notify()
        if pick:
            lines.append(f"🖱️ {pick}")
        selection = self._selection_for_notify()
        if selection:
            lines.append(f"💵 {selection}")
        if self.market_type:
            lines.append(f"Market: {self.market_type}")
        return lines
