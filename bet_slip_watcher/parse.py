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
BET_REF_LOOSE = re.compile(r"\b(CQ[0-9OIl]{8,}[A-Z0-9]*)\b", re.IGNORECASE)
BET_REF_ALT = re.compile(r"\b([0-9A-Z]{2}\d{9,}[A-Z0-9]*)\b")
POINT_SPREAD = re.compile(r"point\s*spread", re.IGNORECASE)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


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


def parse_market_row(
    lines: Iterable[str],
    *,
    prefer_line: str | None = None,
    prefer_odds: str | None = None,
) -> MarketPick | None:
    joined = " ".join(lines)
    text = joined.replace("O", "0")
    candidates: list[MarketPick] = []
    for m in MARKET_ROW.finditer(text):
        candidates.append(
            MarketPick(
                side=m.group(1),
                line=m.group(2),
                odds=m.group(3),
                raw=m.group(0).strip(),
            )
        )
    if not candidates:
        return None
    if prefer_line and prefer_odds:
        want_line = _line_digits(prefer_line)
        for c in candidates:
            if _line_digits(c.line) == want_line and c.odds == prefer_odds:
                return c
    return candidates[0]


def _line_digits(line: str) -> str:
    m = re.search(r"([+-]?\d+\.?\d*)", str(line).replace(" ", ""))
    return m.group(1) if m else line.strip()


def _name_tokens(name: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]", " ", _norm(name))
    return {w for w in cleaned.split() if len(w) >= 3 and not w.isdigit()}


def _selection_matches_team(selection: str, team_name: str) -> bool:
    if not selection or not team_name:
        return False
    st, tt = _name_tokens(selection), _name_tokens(team_name)
    if st & tt:
        return True
    sel_slug = re.sub(r"[^a-z0-9]", "", _norm(selection))
    team_slug = re.sub(r"[^a-z0-9]", "", _norm(team_name))
    if len(sel_slug) >= 5 and len(team_slug) >= 5:
        if sel_slug in team_slug or team_slug in sel_slug:
            return True
    for s in st:
        for t in tt:
            if len(s) >= 4 and len(t) >= 4 and (s in t or t in s or s[:4] == t[:4]):
                return True
    return False


def infer_side_from_event_selection(reading: BetSlipReading) -> str | None:
    """
    bet365 column 1 = first team in 'A vs B', column 2 = second team.
    Do not infer side from +/- on the spread line alone.
    """
    sel = (reading.selection_title or "").strip()
    if not sel:
        return None
    explicit = re.match(r"^(1|2)\b", sel)
    if explicit:
        return explicit.group(1)
    event = reading.event_name or ""
    parts = re.split(r"\s+vs\s+", event, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    team1, team2 = parts[0].strip(), parts[1].strip()
    hit1 = _selection_matches_team(sel, team1)
    hit2 = _selection_matches_team(sel, team2)
    if hit2 and not hit1:
        return "2"
    if hit1 and not hit2:
        return "1"
    if hit1 and hit2:
        s1 = len(_name_tokens(sel) & _name_tokens(team1))
        s2 = len(_name_tokens(sel) & _name_tokens(team2))
        if s2 > s1:
            return "2"
        if s1 > s2:
            return "1"
    return None


def _spread_from_selection(title: str) -> str | None:
    m = re.search(r"([+-]\d+\.?\d*)", title.replace(" ", ""))
    if m:
        return m.group(1)
    m = re.search(r"([+-]?\d+\.?\d*)", _norm(title))
    return m.group(1) if m else None


def reconcile_market_pick(slip: BetSlipReading) -> None:
    """Fix market row side using selection + event (e.g. +12.5 on team 2 is still side 2)."""
    inferred = infer_side_from_event_selection(slip)
    if not inferred:
        return
    mp = slip.market_pick
    line = _spread_from_selection(slip.selection_title or "")
    odds = slip.odds or (mp.odds if mp else None)
    if mp is None and line and odds:
        slip.market_pick = MarketPick(
            side=inferred,
            line=line,
            odds=odds,
            raw=f"{inferred} ({line}) {odds}",
        )
        return
    if mp is None:
        return
    if mp.side != inferred:
        mp.side = inferred
    if line and _line_digits(mp.line) == _line_digits(line):
        mp.line = line
    if odds:
        mp.odds = odds
    mp.raw = f"{mp.side} ({mp.line}) {mp.odds}"


def ocr_lines_indicate_my_bets_settled(lines: Iterable[str]) -> bool:
    """My Bets list (open/settled/live cards), not Place Bet / Bet Placed popup."""
    joined = " ".join(lines).lower()
    if BET_PLACED.search(joined) or PLACE_BET.search(joined):
        return False
    if re.search(r"ret[urn][ne]?d", joined) and "rsd" in joined:
        return True
    if joined.count("single") >= 2:
        return True
    if re.search(r"\b(won|lost)\b", joined) and "single" in joined:
        return True
    if re.search(r"\b(won|lost)\b", joined) and "rsd" in joined:
        return True
    if re.search(r"ret[urn][ne]?d", joined) and "single" in joined:
        return True
    if "settled" in joined and ("open" in joined or "live" in joined):
        return True
    if "to return" in joined and "rsd" in joined:
        return True
    if "reuse" in joined and "selection" in joined:
        return True
    if ("money line" in joined or "point spread" in joined) and (
        "to return" in joined or re.search(r"\bq[1-4]\b", joined)
    ):
        return True
    # Live score table headers (1 2 3 4 T) under bet cards
    if re.search(r"\b1\s+2\b", joined) and re.search(r"\b3\s+4\b", joined):
        return True
    return False


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
            q = m.group(1)
            if q.startswith("bolo") or q.startswith("hol") or q == "boloa":
                return "holon"
            return q
        if re.search(r"bolo", lower):
            return "holon"
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
        title = " ".join(reading.ocr_lines)
    side = infer_side_from_event_selection(reading)
    line = _spread_from_selection(title)
    odds = reading.odds
    if not side and line and odds and reading.selection_title:
        side = "1"
    if side and line and odds:
        return MarketPick(
            side=side,
            line=line,
            odds=odds,
            raw=f"{side} ({line}) {odds}",
        )
    return None


def _normalize_bet_ref(raw: str) -> str:
    ref = raw.upper()
    if ref.startswith(("CQ", "1Q", "KQ")):
        body = ref[2:]
        body = body.replace("O", "0").replace("I", "1").replace("L", "1")
        return ref[:2] + body
    return ref


def _infer_search_from_selection(reading: BetSlipReading) -> str | None:
    title = reading.selection_title or ""
    m = re.search(r"([A-Za-z][A-Za-z\s\-']+?)\s*\(\s*W\s*\)", title)
    if m:
        words = m.group(1).strip().split()
        if words:
            return words[-1].lower()[:16]
    return None


def parse_slip_text(lines: list[str], place_bet_visible: bool) -> BetSlipReading:
    reading = BetSlipReading(
        place_bet_visible=place_bet_visible,
        ocr_lines=list(lines),
    )
    joined = "\n".join(lines)

    if PLACE_BET.search(joined):
        reading.place_bet_visible = True
    if not reading.place_bet_visible and BET_PLACED.search(joined):
        reading.bet_placed_confirmed = True
    elif not reading.place_bet_visible and BET_PLACED_LOOSE.search(joined):
        reading.bet_placed_confirmed = True

    ref = BET_REF.search(joined.replace(" ", ""))
    if not ref:
        ref = BET_REF.search(joined)
    if not ref:
        ref = BET_REF_LOOSE.search(joined.replace(" ", ""))
    if not ref:
        ref = BET_REF_LOOSE.search(joined)
    if not ref:
        ref = BET_REF_ALT.search(joined.replace(" ", ""))
    if ref:
        reading.bet_ref = _normalize_bet_ref(ref.group(1))
        reading.bet_placed_confirmed = True
        reading.place_bet_visible = False

    if POINT_SPREAD.search(joined):
        reading.market_type = "Point Spread"

    odds_hits = ODDS.findall(joined)
    if odds_hits:
        reading.odds = odds_hits[0]

    stake = None
    for line in lines:
        if re.search(r"\bstake\b", line, re.IGNORECASE):
            stake = _parse_rsd(line)
            if stake is not None:
                break
    if stake is None:
        stake = _parse_rsd(joined)
    if stake is None:
        for line in lines:
            low = line.lower()
            if "return" in low and "to" in low:
                continue
            if "rsd" in low or "," in line:
                stake = _parse_rsd(line)
                if stake is not None:
                    break
    if stake is None and not reading.bet_placed_confirmed:
        amounts = LOOSE_AMOUNT.findall(joined)
        best = 0.0
        for raw in amounts:
            val = float(raw.replace(",", ""))
            if val >= 100 and val > best:
                best = val
        if best > 0:
            stake = best

    all_amounts: list[float] = []
    for line in lines:
        for m in RSD_AMOUNT.finditer(line):
            try:
                all_amounts.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
        for raw in LOOSE_AMOUNT.findall(line):
            try:
                val = float(raw.replace(",", ""))
                if val >= 100:
                    all_amounts.append(val)
            except ValueError:
                pass
    uniq = sorted(set(all_amounts))
    if len(uniq) >= 2 and (
        reading.bet_placed_confirmed
        or "return" in joined.lower()
        or sum(1 for _ in RSD_AMOUNT.finditer(joined)) >= 2
    ):
        reading.stake_rsd = uniq[0]
        reading.to_return_rsd = uniq[-1]
    elif stake is not None:
        reading.stake_rsd = stake

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
    if not slip.search_query:
        slip.search_query = _infer_search_from_selection(slip)
    if market_pick is None:
        market_pick = infer_market_from_slip(slip)
    slip.market_pick = market_pick
    if market_pick and market_pick.odds and not slip.odds:
        slip.odds = market_pick.odds
    reconcile_market_pick(slip)
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
