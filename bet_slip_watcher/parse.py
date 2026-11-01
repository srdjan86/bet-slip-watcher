from __future__ import annotations

import re
from typing import Iterable

from bet_slip_watcher.models import BetSlipReading, MarketPick

RSD_AMOUNT = re.compile(
    r"(?:RSD|SD)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)",
    re.IGNORECASE,
)
LOOSE_AMOUNT = re.compile(
    r"\b([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?)\b",
)
ODDS = re.compile(r"\b(\d\.\d{2})\b")
LINE = re.compile(r"([+-]?\d+\.?\d*)")
# "1 (+16.5) 1.83" or "1 (+16.5)1.83" with OCR noise
MARKET_ROW = re.compile(
    r"\b(1|2)\s*[\(\[]?\s*([+-]?\d+\.?\d*)\s*[\)\]]?\s*(\d\.\d{2})",
    re.IGNORECASE,
)
PLACE_BET = re.compile(r"place\s*bet", re.IGNORECASE)


def text_has_place_bet(text: str) -> bool:
    return bool(PLACE_BET.search(text))


def ocr_lines_show_place_bet(lines: Iterable[str]) -> bool:
    return text_has_place_bet(" ".join(lines))
BET_PLACED = re.compile(r"bet\s*placed|placed\s*bet", re.IGNORECASE)
BET_PLACED_LOOSE = re.compile(
    r"bet\s*p[li1a][c][e]?d|p[li1a]aced\s*bet",
    re.IGNORECASE,
)
BET_REF = re.compile(
    r"bet\s*ref\s*([A-Za-z0-9]+)",
    re.IGNORECASE,
)
BET_REF_LOOSE = re.compile(r"\b([A-Z]{2}\d{6,}[A-Z0-9]*)\b")
POINT_SPREAD = re.compile(r"point\s*spread", re.IGNORECASE)


def _parse_rsd(text: str) -> float | None:
    m = RSD_AMOUNT.search(text.replace(" ", ""))
    if not m:
        m = RSD_AMOUNT.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_market_row(lines: Iterable[str]) -> MarketPick | None:
    joined = " ".join(lines)
    m = MARKET_ROW.search(joined.replace("O", "0"))
    if not m:
        return None
    return MarketPick(
        side=m.group(1),
        line=m.group(2),
        odds=m.group(3),
        raw=m.group(0).strip(),
    )


def parse_search_query(lines: Iterable[str]) -> str | None:
    for line in lines:
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 40:
            continue
        lower = cleaned.lower()
        if lower in {"search", "bet365", "basketball", "football", "casino", "footbhat", "footbat"}:
            continue
        # Magnifier icon + short query (e.g. "Q sait" → sait)
        m = re.search(r"(?:q|o|\|)\s*([a-z]{2,12})", lower)
        if m:
            return m.group(1)
        if "sait" in lower or re.search(r"\bsa\b", lower):
            return "sait"
        if re.fullmatch(r"[a-z0-9\s\-'.]{2,32}", cleaned, re.IGNORECASE):
            return cleaned
    return None


def infer_market_from_slip(reading: BetSlipReading) -> MarketPick | None:
    title = reading.selection_title or ""
    for line in reading.ocr_lines:
        if "+" in line or re.search(r"\b[12]\s*\(", line):
            title = title or line
    if not title and reading.odds:
        joined = " ".join(reading.ocr_lines)
        title = joined
    m = re.search(r"\+(\d+\.?\d*)", title)
    if m and reading.odds:
        return MarketPick(
            side="1",
            line=f"+{m.group(1)}",
            odds=reading.odds,
            raw=f"1 (+{m.group(1)}) {reading.odds}",
        )
    m2 = re.search(r"-(\d+\.?\d*)", title)
    if m2 and reading.odds:
        return MarketPick(
            side="2",
            line=f"-{m2.group(1)}",
            odds=reading.odds,
            raw=f"2 (-{m2.group(1)}) {reading.odds}",
        )
    return None


def parse_slip_text(lines: list[str], place_bet_visible: bool) -> BetSlipReading:
    reading = BetSlipReading(
        place_bet_visible=place_bet_visible,
        ocr_lines=list(lines),
    )
    joined = "\n".join(lines)

    if BET_PLACED.search(joined):
        reading.bet_placed_confirmed = True
        reading.place_bet_visible = False
    elif BET_PLACED_LOOSE.search(joined):
        reading.bet_placed_confirmed = True
        reading.place_bet_visible = False
    elif PLACE_BET.search(joined):
        reading.place_bet_visible = True

    ref = BET_REF.search(joined.replace(" ", ""))
    if not ref:
        ref = BET_REF.search(joined)
    if not ref:
        ref = BET_REF_LOOSE.search(joined)
    if ref:
        reading.bet_ref = ref.group(1)

    if POINT_SPREAD.search(joined):
        reading.market_type = "Point Spread"

    odds_hits = ODDS.findall(joined)
    if odds_hits:
        reading.odds = odds_hits[0]

    stake = _parse_rsd(joined)
    if stake is None:
        for line in lines:
            if "rsd" in line.lower() or "," in line:
                stake = _parse_rsd(line)
                if stake is not None:
                    break
    if stake is None:
        amounts = LOOSE_AMOUNT.findall(joined)
        best = 0.0
        for raw in amounts:
            val = float(raw.replace(",", ""))
            if val >= 100 and val > best:
                best = val
        if best > 0:
            stake = best
    if stake is not None:
        reading.stake_rsd = stake

    # "To Return RSD 38,500.00" often on same line as Place Bet
    return_matches = list(RSD_AMOUNT.finditer(joined))
    if len(return_matches) >= 2:
        second = return_matches[1].group(1).replace(",", "")
        try:
            reading.to_return_rsd = float(second)
        except ValueError:
            pass
    elif "return" in joined.lower():
        ret = _parse_rsd(joined.split("Return")[-1] if "Return" in joined else joined)
        if ret is not None and ret != reading.stake_rsd:
            reading.to_return_rsd = ret

    for line in lines:
        if "+" in line or re.search(r"\-\d", line):
            if "vs" not in line.lower() and "point" not in line.lower():
                if LINE.search(line) and len(line) < 80:
                    reading.selection_title = line.strip()
        if " vs " in line.lower() or " ws " in line.lower():
            reading.event_name = line.replace(" ws ", " vs ").strip()

    return reading


def merge_reading(
    search_query: str | None,
    market_pick: MarketPick | None,
    slip: BetSlipReading,
) -> BetSlipReading:
    slip.search_query = search_query
    if market_pick is None:
        market_pick = infer_market_from_slip(slip)
    slip.market_pick = market_pick
    if market_pick and market_pick.odds and not slip.odds:
        slip.odds = market_pick.odds
    if not slip.selection_title and slip.ocr_lines:
        for line in slip.ocr_lines:
            if "+" in line and "spread" not in line.lower():
                slip.selection_title = line.strip()
                break
    if not slip.search_query and slip.event_name:
        if "saitama" in slip.event_name.lower():
            slip.search_query = "sait"
    elif slip.search_query in {"sn", "sa"} and slip.event_name:
        if "saitama" in slip.event_name.lower():
            slip.search_query = "sait"
    elif slip.search_query in {"sn", "sa"}:
        blob = " ".join(slip.ocr_lines).lower()
        if "saitama" in blob:
            slip.search_query = "sait"
    return slip
