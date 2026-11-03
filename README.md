# Bet slip stream watcher (Phase 1)

Watches a **1080p desktop stream** of bet365.rs, detects the bottom **bet slip** (green **Place Bet** + stake), and sends a notification with the inferred path your friend took:

1. **Search** (e.g. `sait` in the top search bar)  
2. **Click** on the coupon row (e.g. `1 (+16.5) 1.83`)  
3. **Stake** + **Place Bet** on the slip → optional **bet likely placed** when the slip disappears

No bet365 login or auto-betting in this phase.

## Setup

```bash
cd ~/bet-slip-watcher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# streamlink + Tesseract for OCR
brew install streamlink tesseract yt-dlp ffmpeg   # macOS
cp .env.example .env
# edit .env — STREAM_URL, Firefox cookies (below), optional TELEGRAM_* or NTFY_TOPIC
```

Use **Python 3.11–3.12** if EasyOCR is added later; this sketch uses **Tesseract** via `pytesseract`.

## Tune on your screenshot

Use the frame you shared before going live:

```bash
python -m bet_slip_watcher.main \
  --image "/path/to/your/screenshot.png" \
  --verbose \
  --debug-save-crops ./crops
```

Check `./crops/search_0.png`, `market_0.png`, `slip_0.png`. Adjust `regions` and `place_bet_green_hsv` in `config.yaml` until the crops tightly frame the search text, odds row, and slip.

Expected console fields at verbose:

- `search=sait`
- `pick=1 (+16.5) 1.83` (approx)
- `stake=21000.0`

## Run on YouTube

```bash
python -m bet_slip_watcher.main --verbose
```

**Rewind and troubleshoot missed Telegram alerts** (watch from **4 hours before now**, save every screenshot):

```bash
python -m bet_slip_watcher.main --verbose --rewind 4h15m
```

(Also: `--rewind 4:15:00`, `--rewind 4h`, `--rewind 1h`.)

`--rewind` uses yt-dlp **`--download-sections *-15300-inf`** (seconds **before the live edge**, not the scrub bar). That matters on long streams where “stream start” is days ago but DVR only keeps ~12h.

Optional **`--max-frames 500`** to cap a test run. Durations: `1h`, `4h`, `30m`, or `H:MM:SS`.

**Live** (no rewind):

```bash
python -m bet_slip_watcher.main --verbose
```

**Advanced:** scrub-bar time since stream start (if you know the exact timestamp on YouTube):

```bash
python -m bet_slip_watcher.main --verbose --stream-from 4:13:00 --audit-dir ./audit
```

Watch the log for `Notify bet_placing (total 1)…` through `#4` and matching Telegram messages. Each distinct wager (event + selection + line + odds + stake) notifies **once**; state resets when the slip UI clears between bets.

**Audit frames** — filenames like `000042_not-placed_bet-placing_not-ready_slip-on-screen_stake21000_ref-no-ref.png`  
(`not-placed` / `not-placing` on most frames is normal; scroll for moments where the UI shows a bet but labels stay `not-*`).  
Telegram sends also get extra `NOTIFY-bet_placing_…` copies in the same folder.

### Private YouTube (invite-only viewers)

The watcher uses **yt-dlp** with your Google session. On a **Mac**, use a **dedicated Firefox profile** for the stream account only (separate from your daily Chrome/Safari).

#### Recommended: Firefox “Stream” profile + live cookies (no export)

1. Install **Firefox** (only for this if you like — keeps stream login isolated from Chrome).
2. **Firefox → `about:profiles` → Create profile** (e.g. name **Stream**). Launch that profile and sign in with the Google account that can watch the live stream. Open the live URL once and confirm it plays.
3. Note the profile **folder name** (not the display name). On the profiles page, **Root Directory** ends with something like:

   `…/Firefox/Profiles/skqlrte6.Stream`

   yt-dlp needs the last segment: **`skqlrte6.Stream`** (yours will differ).

   Or in Terminal:

   ```bash
   grep -E '^(Name|Path)=' ~/Library/Application\ Support/Firefox/profiles.ini
   ```

4. In `.env`:

   ```bash
   STREAM_URL=https://www.youtube.com/watch?v=YOUR_VIDEO_ID
   YTDLP_COOKIES_FROM_BROWSER="firefox:skqlrte6.Stream"
   ```

   Replace with **your** folder name (case-sensitive).

5. Test:

   ```bash
   yt-dlp --simulate --cookies-from-browser "firefox:skqlrte6.Stream" \
     'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
   ```

   Expect `Extracted … cookies from firefox` and `[info] … Downloading 1 format(s): …` — not `Video unavailable`.

6. Run the watcher. yt-dlp reads **that profile’s** cookie DB on each connect/reconnect — no manual `cookies.txt`, and **no cookies from other browsers or Firefox profiles**.

Keep the stream profile logged in on YouTube. Quit Firefox if cookie read fails once, then retry.

#### Alternative: dedicated Chrome profile

Same idea with `YTDLP_COOKIES_FROM_BROWSER="chrome:Profile 3"` (folder name from `chrome://version` → Profile Path). Use a Chrome profile that is **not** your main daily profile.

#### Fallback: manual `cookies.txt` export

Export YouTube-only cookies on the watch page to `secrets/youtube_cookies.txt` and set `STREAMLINK_HTTP_COOKIE_FILE=…` in `.env`. Re-export when playback fails.

Notifications:

- **slip_ready** — stake filled, Place Bet visible (off by default; `notify_slip_ready: true` in config.yaml)
- **bet_placing** — first frame with spinner (fastest alert)
- **bet_placed** — if placing was missed, **Bet Placed** on the **next frame** (same wager still only one message)
- **bet_likely_placed** — last resort if both OCR steps miss but the slip vanishes after a wager was tracked

## Telegram

1. Create a bot via [@BotFather](https://t.me/BotFather), set `TELEGRAM_BOT_TOKEN`
2. Message the bot, then resolve chat id (e.g. `https://api.telegram.org/bot<TOKEN>/getUpdates`)
3. Set `TELEGRAM_CHAT_ID` in `.env`

Or set `NTFY_TOPIC=my-secret-topic` and subscribe in the ntfy app.

## Project layout

```
bet_slip_watcher/
  main.py          CLI loop
  stream.py        streamlink + OpenCV capture
  regions.py       fractional crops for 1080p layout
  detect.py        green Place Bet + dark slip heuristics
  ocr_engine.py    Tesseract wrapper
  parse.py         search / market row / RSD stake parsing
  state.py         slip_ready vs bet_likely_placed
  notify.py        Telegram + ntfy
  models.py        datatypes
config.yaml        regions + HSV thresholds
```

## Next phase (not implemented)

Structured JSON output to drive **your** browser: search `{{search_query}}`, open line `{{side}} ({{line}}) @ {{odds}}`, stake cap, manual confirm — via Playwright or local screen automation.
