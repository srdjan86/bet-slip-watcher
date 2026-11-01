from __future__ import annotations

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
    ocr_lines: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.search_query:
            lines.append(f"Search: {self.search_query}")
        if self.market_pick and self.market_pick.raw:
            lines.append(f"Clicked row: {self.market_pick.raw}")
        elif self.market_pick and self.market_pick.side:
            mp = self.market_pick
            lines.append(f"Clicked: {mp.side} ({mp.line}) @ {mp.odds}")
        if self.selection_title:
            lines.append(f"Selection: {self.selection_title}")
        if self.market_type:
            lines.append(f"Market: {self.market_type}")
        if self.event_name:
            lines.append(f"Event: {self.event_name}")
        if self.odds:
            lines.append(f"Odds: {self.odds}")
        if self.stake_rsd is not None:
            lines.append(f"Stake: RSD {self.stake_rsd:,.2f}")
        if self.to_return_rsd is not None:
            lines.append(f"To return: RSD {self.to_return_rsd:,.2f}")
        if self.bet_ref:
            lines.append(f"Bet ref: {self.bet_ref}")
        return lines
