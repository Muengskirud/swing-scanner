"""
SkimTrade (Streamlit). A research screener for oversold rebounds after emotional
sell-offs: finds US stocks at the lower Bollinger Band with low RSI and heavy volume,
builds a trade plan from a fixed dollar risk and reward:risk (minimum 1:2), and shows
the headlines behind each drop.

Setup:
    pip install -r requirements.txt
    streamlit run swing_app.py

Analysis only. Nothing here connects to a broker or places orders.
"""
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from tradingview_screener import Query, col

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "settings.json")
SCANS_DIR = os.path.join(HERE, "scans")
# Hosted (e.g. Streamlit Community Cloud): visitors share one server, so nothing is written to disk.
HOSTED = os.environ.get("SWING_HOSTED") == "1" or HERE.startswith("/mount/src")
ET_TZ = ZoneInfo("America/New_York")
SG_TZ = ZoneInfo("Asia/Singapore")
MIN_RR = 2.0  # strategy floor: never plan below 1:2
PREMARKET_GAP = -3.0  # premarket move (%) that counts as an overnight shock
FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
FG_PAGE = "https://www.cnn.com/markets/fear-and-greed"

COLUMNS = [
    "name", "description", "close", "change", "RSI", "BB.lower", "BB.upper", "ATR",
    "relative_volume_10d_calc", "average_volume_10d_calc", "market_cap_basic",
    "sector", "industry", "earnings_release_next_date", "earnings_release_date",
    "SMA200", "Perf.W", "premarket_close", "premarket_change", "update_mode",
]
DEFAULTS = {
    "layman": True, "max_risk": 35, "rr": 2.0, "stop_mode": "ATR", "atr_mult": 1.5,
    "target_pct": 4.0, "max_hold": 7,
    "min_mcap_b": 2.0, "min_price": 5.0, "min_avg_vol": 1_000_000, "rsi_min": 20, "rsi_max": 35,
    "min_rel_vol": 1.5, "bb_tol": 1.01, "earn_buffer": 7,
}

SEVERE = re.compile(
    r"\b(fraud|investigation|subpoena|bankruptcy|chapter 11|going concern|restatement|"
    r"delist\w*|short report|sec charges|accounting irregularit\w*)\b", re.I)
CAUTION = re.compile(
    r"\b(downgrad\w*|lawsuit|sues|recall\w*|offering|resign\w*|layoffs?|probe|"
    r"cuts? guidance|lowers? guidance|misses?)\b", re.I)


# ---------------------------------------------------------------- settings
def load_settings() -> dict:
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save_settings(p: dict):
    with open(SETTINGS, "w", encoding="utf-8") as f:
        json.dump({k: p[k] for k in DEFAULTS}, f, indent=2)


# ---------------------------------------------------------------- market + scan
def market_session(now: datetime | None = None) -> str:
    """US equity session by New York clock. Exchange holidays are not handled."""
    now = (now or datetime.now(timezone.utc)).astimezone(ET_TZ)
    if now.weekday() >= 5:
        return "closed (weekend)"
    t = now.hour * 60 + now.minute
    if t < 4 * 60:
        return "closed"
    if t < 9 * 60 + 30:
        return "premarket"
    if t < 16 * 60:
        return "open"
    if t < 20 * 60:
        return "after hours"
    return "closed"


def next_session_event(now: datetime | None = None) -> tuple[str, datetime]:
    """What changes next and when: premarket 4:00, open 9:30, close 16:00, after-hours end 20:00
    (New York time). Weekends skip to Monday premarket. Exchange holidays are not handled."""
    n = (now or datetime.now(timezone.utc)).astimezone(ET_TZ)

    def at(d, h, m):
        return datetime(d.year, d.month, d.day, h, m, tzinfo=ET_TZ)

    t = n.hour * 60 + n.minute
    if n.weekday() < 5:
        for end, label, h, m in [(240, "Premarket opens", 4, 0), (570, "Market opens", 9, 30),
                                 (960, "Market closes", 16, 0), (1200, "After-hours closes", 20, 0)]:
            if t < end:
                return label, at(n, h, m)
    d = n.date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return "Premarket opens", at(d, 4, 0)


def countdown(target: datetime, now: datetime | None = None) -> str:
    mins = max(0, int((target - (now or datetime.now(timezone.utc))).total_seconds() // 60))
    d, h, m = mins // 1440, mins % 1440 // 60, mins % 60
    return f"{d}d {h}h" if d else f"{h}h {m}m" if h else f"{m}m"


def _base_filters(p: dict) -> list:
    return [
        col("type") == "stock",  # excludes ETFs, including leveraged ones
        col("market_cap_basic") > p["min_mcap_b"] * 1e9,
        col("close") > p["min_price"],
        col("average_volume_10d_calc") > p["min_avg_vol"],
    ]


def scan(p: dict) -> pd.DataFrame:
    """Oversold stocks trading on heavy volume in the latest session. The lower-band and
    RSI-floor checks run afterwards so names that fail them can be shown with the reason."""
    q = (Query().set_markets("america").select(*COLUMNS)
         .where(*_base_filters(p),
                col("RSI") < p["rsi_max"],
                col("relative_volume_10d_calc") > p["min_rel_vol"])
         .order_by("RSI", ascending=True).limit(500))
    _, df = q.get_scanner_data()
    df["source"] = "sell-off"
    return df


def scan_premarket_gaps(p: dict) -> pd.DataFrame:
    """Overnight shocks: names gapping down in premarket. Indicators are still from the last close."""
    q = (Query().set_markets("america").select(*COLUMNS)
         .where(*_base_filters(p), col("premarket_change") < PREMARKET_GAP)
         .order_by("premarket_change", ascending=True).limit(50))
    _, df = q.get_scanner_data()
    df["source"] = "premarket gap"
    return df


def run_scan(p: dict, session: str) -> pd.DataFrame:
    frames = [scan(p)]
    if session == "premarket":
        frames.append(scan_premarket_gaps(p))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("name", keep="first")
    if not HOSTED:
        os.makedirs(SCANS_DIR, exist_ok=True)
        df.to_csv(os.path.join(SCANS_DIR, f"scan_{datetime.now():%Y%m%d_%H%M}.csv"), index=False)
    return df


def ref_price(row, session: str) -> float:
    pm = row.get("premarket_close")
    if session == "premarket" and pm is not None and not pd.isna(pm):
        return float(pm)
    return float(row["close"])


# ---------------------------------------------------------------- plan + checks
def build_plan(entry: float, atr: float, p: dict) -> dict:
    """Size purely from the dollar risk: shares = max loss / (entry - stop)."""
    if p["stop_mode"] == "ATR":
        stop_dist = p["atr_mult"] * atr
    else:
        stop_dist = entry * p["target_pct"] / 100 / p["rr"]
    shares = math.floor(p["max_risk"] / stop_dist) if stop_dist > 0 else 0
    return {
        "entry": round(entry, 2),
        "stop": round(entry - stop_dist, 2),
        "target": round(entry + p["rr"] * stop_dist, 2),
        "stop_dist": stop_dist,
        "stop_pct": round(100 * stop_dist / entry, 2) if entry else 0,
        "target_pct": round(100 * p["rr"] * stop_dist / entry, 2) if entry else 0,
        "stop_atr": round(stop_dist / atr, 2) if atr else 0,
        "shares": shares,
        "position": round(shares * entry, 2),
        "risk": round(shares * stop_dist, 2),
    }


def days_from_now(ts):
    if ts is None or pd.isna(ts):
        return None
    return (datetime.fromtimestamp(float(ts), tz=timezone.utc) - datetime.now(timezone.utc)).days


def checks(row, plan, news, p) -> list:
    """Automatic pre-checks. Each is (pass label, failure in plain words, ok).
    These can only rule things out, never confirm a thesis."""
    ed = days_from_now(row["earnings_release_next_date"])
    levels = [x["level"] for x in news]
    bb_line = row["BB.lower"] * p["bb_tol"]
    return [
        ("At or below lower Bollinger Band",
         f"price is {row['bb_dist_pct']:+.1f}% vs the lower band (limit ${bb_line:.2f})",
         plan["entry"] <= bb_line),
        (f"RSI between {p['rsi_min']} and {p['rsi_max']}",
         (f"RSI {row['RSI']:.0f} is below {p['rsi_min']}: falling-knife zone, check the news hard"
          if row["RSI"] < p["rsi_min"] else f"RSI {row['RSI']:.0f} is not under {p['rsi_max']}"),
         p["rsi_min"] <= row["RSI"] < p["rsi_max"]),
        (f"Volume at least {p['min_rel_vol']}x average",
         f"volume only {row['relative_volume_10d_calc']:.1f}x average",
         row["relative_volume_10d_calc"] >= p["min_rel_vol"]),
        (f"No earnings within {p['earn_buffer']} days" + (" (date unknown)" if ed is None else ""),
         f"earnings in {ed} days (gap risk)",
         ed is None or not (0 <= ed <= p["earn_buffer"])),
        ("No severe red-flag headline", "severe red-flag headline", "severe" not in levels),
        ("At least 1 share fits your max loss",
         f"stop is wider than your ${p['max_risk']:g} max loss", plan["shares"] >= 1),
    ]


# ---------------------------------------------------------------- news
def parse_news_xml(content: bytes, n: int = 6) -> list:
    root = ET.fromstring(content)
    items = []
    for it in root.iter("item"):
        title = it.findtext("title") or ""
        items.append({
            "title": title,
            "link": it.findtext("link") or "",
            "source": it.findtext("source") or "",
            "published": (it.findtext("pubDate") or "")[5:16],
            "level": "severe" if SEVERE.search(title) else "caution" if CAUTION.search(title) else "ok",
        })
        if len(items) >= n:
            break
    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker: str, company: str, n: int = 6) -> list:
    q = quote(f"{company} {ticker} stock when:7d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return parse_news_xml(r.content, n)
    except Exception as e:  # network or parse failure: say so, never fake it
        return [{"title": f"News unavailable ({type(e).__name__})", "link": "", "source": "",
                 "published": "", "level": "error"}]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fear_greed() -> dict | None:
    """CNN Fear & Greed Index. CNN rejects requests that don't look like a browser."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
               "Accept": "application/json", "Referer": "https://www.cnn.com/", "Origin": "https://www.cnn.com"}
    try:
        r = requests.get(FG_URL, headers=headers, timeout=10)
        r.raise_for_status()
        fg = r.json()["fear_and_greed"]
        return {"score": float(fg["score"]), "rating": str(fg["rating"]).lower(),
                "prev": fg.get("previous_close"), "week": fg.get("previous_1_week"),
                "asof": str(fg.get("timestamp", ""))[:10]}
    except Exception:  # never fake it: the tile says unavailable and still links to CNN
        return None


# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #0B0F1A; --panel: #121829; --panel2: #182036; --line: #232C45;
  --text: #E8EBF5; --muted: #8A93AD; --violet: #8B6CFF; --violet2: #5B8CFF;
  --green: #2BD9A0; --red: #FF5C7A; --amber: #FFB547;
}
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2, .stApp h3,
.stApp li, .stApp td, .stApp th, .stMarkdown, .sw { font-family: 'Montserrat', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }

.sw-hero { background: radial-gradient(1200px 300px at 0% 0%, rgba(139,108,255,.35), transparent 60%),
           radial-gradient(800px 260px at 100% 0%, rgba(43,217,160,.18), transparent 60%), var(--panel);
           border: 1px solid var(--line); border-radius: 22px; padding: 28px 30px; margin-bottom: 18px; }
.sw-hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; color: var(--text); }
.sw-hero .sub { color: var(--muted); font-weight: 500; margin-top: 4px; font-size: .95rem; }
.sw-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.sw-pill { background: rgba(255,255,255,.06); border: 1px solid var(--line); border-radius: 999px;
           padding: 6px 12px; font-size: .8rem; font-weight: 600; color: var(--text); }
.sw-pill.live { border-color: rgba(43,217,160,.5); color: var(--green); }
.sw-pill.pre  { border-color: rgba(255,181,71,.5); color: var(--amber); }
.sw-pill.off  { border-color: rgba(255,92,122,.4); color: #FF9AAE; }
.sw-pill.cd   { background: rgba(139,108,255,.16); border-color: rgba(139,108,255,.55); }
.sw-pill.cd b { font-weight: 800; }

.sw-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 22px; }
.sw-stat { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; }
.sw-stat .n { font-size: 1.7rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.sw-stat .l { font-size: .75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; }

.sw-section { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 8px 0 10px; }

.sw-card { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 20px;
           margin-bottom: 6px; transition: border-color .2s, transform .2s; }
.sw-card:hover { border-color: rgba(139,108,255,.6); transform: translateY(-2px); }
.sw-card.skip { opacity: .72; }
.sw-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.sw-tkr { font-size: 1.55rem; font-weight: 800; color: var(--text); letter-spacing: -.3px; }
.sw-co { color: var(--muted); font-size: .8rem; font-weight: 500; margin-top: 2px; }
.sw-chip { border-radius: 999px; padding: 5px 11px; font-size: .75rem; font-weight: 700; white-space: nowrap; }
.sw-chip.go { background: rgba(43,217,160,.14); color: var(--green); border: 1px solid rgba(43,217,160,.35); }
.sw-chip.no { background: rgba(255,92,122,.12); color: var(--red); border: 1px solid rgba(255,92,122,.35); }

.sw-mini { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 16px 0 14px; }
.sw-mini > div { background: var(--panel2); border-radius: 12px; padding: 9px 10px; }
.sw-mini .k { font-size: .66rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.sw-mini .v { font-size: 1rem; font-weight: 700; color: var(--text); margin-top: 2px; }
.red { color: var(--red) !important; } .green { color: var(--green) !important; } .amber { color: var(--amber) !important; }
.sw-meter { height: 5px; background: rgba(255,255,255,.08); border-radius: 9px; margin-top: 6px; overflow: hidden; }
.sw-meter span { display: block; height: 100%; border-radius: 9px; }

.sw-plan { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.sw-plan > div { border-radius: 14px; padding: 10px 12px; min-width: 0; }
.sw-plan .e { background: rgba(139,108,255,.14); border: 1px solid rgba(139,108,255,.35); }
.sw-plan .s { background: rgba(255,92,122,.10); border: 1px solid rgba(255,92,122,.30); }
.sw-plan .t { background: rgba(43,217,160,.10); border: 1px solid rgba(43,217,160,.30); }
.sw-plan .k { font-size: .7rem; font-weight: 700; color: var(--muted); }
.sw-plan .v { font-size: 1.1rem; white-space: nowrap; font-weight: 800; color: var(--text); }
.sw-plan .d { font-size: .72rem; font-weight: 600; white-space: nowrap; }

.sw-ladder { position: relative; height: 8px; border-radius: 9px; margin: 14px 2px 4px;
             background: linear-gradient(90deg, var(--red) 0%, var(--red) var(--cut), var(--green) var(--cut), var(--green) 100%); }
.sw-ladder i { position: absolute; top: -5px; left: var(--cut); width: 4px; height: 18px; margin-left: -2px;
               background: #fff; border-radius: 3px; box-shadow: 0 0 10px rgba(139,108,255,.9); }
.sw-foot { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; font-size: .8rem;
           font-weight: 600; color: var(--muted); margin-top: 12px; }
.sw-foot b { color: var(--text); }
.sw-why { margin-top: 12px; background: rgba(255,92,122,.08); border-left: 3px solid var(--red);
          border-radius: 10px; padding: 9px 12px; font-size: .8rem; color: #FFC2CE; font-weight: 500; }
.sw-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.sw-tag { font-size: .72rem; font-weight: 600; padding: 4px 9px; border-radius: 8px;
          background: rgba(255,181,71,.10); color: var(--amber); border: 1px solid rgba(255,181,71,.25); }

.sw-news a { color: var(--text) !important; text-decoration: none; font-weight: 600; font-size: .85rem; }
.sw-news a:hover { color: var(--violet) !important; }
.sw-news .src { color: var(--muted); font-size: .72rem; margin: 2px 0 10px; }
.sw-check { font-size: .82rem; font-weight: 500; margin: 3px 0; }

div[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: 14px !important;
                                        background: var(--panel) !important; }
div.stButton > button[kind="primary"] { background: linear-gradient(90deg, var(--violet), var(--violet2));
    border: 0; border-radius: 14px; font-weight: 700; padding: .65rem 1rem; font-size: 1rem;
    box-shadow: 0 8px 30px rgba(139,108,255,.35); }
section[data-testid="stSidebar"] { background: #0E1322; border-right: 1px solid var(--line); }
.sw-hero { display: grid; grid-template-columns: 1fr 270px; gap: 22px; align-items: center; }
.sw-fg { background: rgba(11,15,26,.55); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px;
         cursor: help; }
.sw-fg .k { font-size: .7rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; }
.sw-fg .n { font-size: 2.4rem; font-weight: 800; line-height: 1.05; margin-top: 4px; }
.sw-fg .r { font-size: .85rem; font-weight: 700; text-transform: capitalize; }
.sw-fg .bar { position: relative; height: 8px; border-radius: 9px; margin: 12px 0 8px;
              background: linear-gradient(90deg, #2BD9A0 0 25%, #8EE6A8 25% 45%, #FFC94D 45% 55%, #FF5C7A 55% 75%, #B3243B 75% 100%); }
.sw-fg .bar i { position: absolute; top: -4px; width: 4px; height: 16px; margin-left: -2px; background: #fff;
                border-radius: 3px; box-shadow: 0 0 8px rgba(255,255,255,.8); }
.sw-fg .m { font-size: .72rem; color: var(--muted); font-weight: 600; display: flex; justify-content: space-between; }
.sw-fg a { color: var(--violet) !important; text-decoration: none; font-weight: 700; }
/* hover bubbles */
[data-tip] { position: relative; cursor: help; }
[data-tip]:hover::after {
  content: attr(data-tip); position: absolute; z-index: 1000; left: 50%; bottom: calc(100% + 12px);
  transform: translateX(-50%); width: 240px; min-height: 150px; box-sizing: border-box;
  display: flex; align-items: center; padding: 16px 18px; border-radius: 18px;
  background: linear-gradient(160deg, #241C4A 0%, #182036 70%); border: 1px solid rgba(139,108,255,.55);
  box-shadow: 0 18px 40px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.03);
  color: var(--text); font-family: 'Montserrat', sans-serif; font-size: .78rem; font-weight: 500;
  line-height: 1.5; letter-spacing: 0; text-transform: none; white-space: normal; text-align: left;
  pointer-events: none; animation: swpop .15s ease-out; }
[data-tip]:hover::before {
  content: ""; position: absolute; z-index: 1001; left: 50%; bottom: calc(100% + 5px); transform: translateX(-50%);
  border: 7px solid transparent; border-top-color: rgba(139,108,255,.55); pointer-events: none; }
.sw-fg[data-tip]:hover::after { bottom: auto; top: calc(100% + 12px); left: auto; right: 0; transform: none; }
.sw-fg[data-tip]:hover::before { bottom: auto; top: calc(100% - 2px); left: auto; right: 40px; transform: none;
  border-top-color: transparent; border-bottom-color: rgba(139,108,255,.55); }
@keyframes swpop { from { opacity: 0; } to { opacity: 1; } }
.sw-card { position: relative; } .sw-card:hover { z-index: 20; }
.sw-hero { position: relative; z-index: 30; } .sw-stats { position: relative; z-index: 25; }
/* Streamlit's own (?) help bubbles in the sidebar */
div[data-testid="stTooltipContent"], div[data-baseweb="tooltip"] > div {
  max-width: 260px !important; border-radius: 18px !important; padding: 16px 18px !important;
  background: linear-gradient(160deg, #241C4A 0%, #182036 70%) !important;
  border: 1px solid rgba(139,108,255,.55) !important; box-shadow: 0 18px 40px rgba(0,0,0,.55) !important; }
div[data-testid="stTooltipContent"] p { font-family: 'Montserrat', sans-serif !important; font-size: .8rem !important;
  line-height: 1.5 !important; color: var(--text) !important; }
/* sidebar labels with a (?) whose bubble opens to the right, over the empty main area */
.sw-lbl { display: flex; justify-content: space-between; align-items: center; font-size: .86rem;
          font-weight: 600; color: var(--text); margin: 8px 0 4px; }
.sw-q { display: inline-block; text-align: center; line-height: 14px; width: 17px; height: 17px; box-sizing: border-box;
        border-radius: 50%; border: 1.5px solid var(--muted); color: var(--muted); font-size: .66rem;
        font-weight: 800; flex: none; transition: all .15s; }
.sw-q:hover { border-color: var(--violet); color: #fff; background: var(--violet); }
.sw-q[data-tip]:hover::after, .sw-q[data-tip]:hover::before { display: none; }
.sw-float { position: fixed; z-index: 2147483000; width: 260px; min-height: 150px; box-sizing: border-box;
  display: flex; align-items: center; padding: 16px 18px; border-radius: 18px; pointer-events: none;
  background: linear-gradient(160deg, #241C4A 0%, #182036 70%); border: 1px solid rgba(139,108,255,.55);
  box-shadow: 0 18px 40px rgba(0,0,0,.55); color: #E8EBF5; font-family: 'Montserrat', sans-serif;
  font-size: .78rem; font-weight: 500; line-height: 1.5; animation: swpop .15s ease-out; }
div[data-testid="stElementContainer"]:has(iframe[height="0"]) { display: none; }
.sw-float::before { content: ""; position: absolute; left: -15px; top: var(--ay, 24px); margin-top: -7px;
  border: 7px solid transparent; border-right-color: rgba(139,108,255,.75); }
.st-key-layman { opacity: .6; transition: opacity .15s; }
.st-key-layman:hover { opacity: 1; }
.st-key-layman label p { font-size: .75rem !important; font-weight: 600; }
.sw-tkr a { color: var(--text) !important; text-decoration: none; }
.sw-tkr a:hover { color: var(--violet) !important; }
.sw-tkr a span { font-size: .8rem; color: var(--muted); margin-left: 4px; }
@media (max-width: 760px) { .sw-stats, .sw-mini { grid-template-columns: repeat(2, 1fr); }
                            .sw-hero { grid-template-columns: 1fr; } }
</style>
"""


def money(x: float) -> str:
    return f"${x:,.2f}"


# ---------------------------------------------------------------- UI: sidebar
TIP_JS = """<script>
(function () {
  const doc = window.parent.document;
  if (window.parent.__swTips) return;
  window.parent.__swTips = true;
  let tip = null;
  doc.addEventListener("mouseover", (e) => {
    const q = e.target.closest && e.target.closest(".sw-q");
    if (!q) return;
    if (!tip) { tip = doc.createElement("div"); tip.className = "sw-float"; doc.body.appendChild(tip); }
    tip.textContent = q.getAttribute("data-tip");
    tip.style.display = "flex";
    const r = q.getBoundingClientRect(), h = tip.offsetHeight, vh = window.parent.innerHeight;
    const cy = r.top + r.height / 2;
    const top = Math.max(12, Math.min(cy - 28, vh - h - 12));
    const sb = q.closest('section[data-testid="stSidebar"]');
    const edge = sb ? sb.getBoundingClientRect().right : r.right;
    tip.style.left = (Math.max(r.right, edge) + 18) + "px";
    tip.style.top = top + "px";
    tip.style.setProperty("--ay", (cy - top) + "px");
  });
  const fmt = (ms) => {
    const m = Math.max(0, Math.floor(ms / 60000)), d = Math.floor(m / 1440), h = Math.floor(m % 1440 / 60);
    return ms <= 0 ? "now, refresh" : d ? d + "d " + h + "h" : h ? h + "h " + (m % 60) + "m" : (m % 60) + "m";
  };
  const tick = () => {
    doc.querySelectorAll(".sw-cd").forEach((e) => { e.textContent = fmt(+e.dataset.until - Date.now()); });
    doc.querySelectorAll(".sw-clock").forEach((e) => {
      e.textContent = new Intl.DateTimeFormat("en-GB", { timeZone: e.dataset.tz, weekday: "short",
        hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date()).replace(",", "");
    });
  };
  window.parent.setInterval(tick, 15000);
  doc.addEventListener("mouseout", (e) => {
    const q = e.target.closest && e.target.closest(".sw-q");
    if (q && tip) tip.style.display = "none";
  });
})();
</script>"""


def field(where, kind: str, label: str, *args, help: str = "", **kw):
    """Sidebar input with our own label and (?) bubble, which opens to the right of the
    sidebar instead of covering the inputs like Streamlit's built-in help tooltip."""
    q = f'<span class="sw-q" data-tip="{html.escape(help)}">?</span>' if help else ""
    where.markdown(f'<div class="sw sw-lbl"><span>{html.escape(label)}</span>{q}</div>',
                   unsafe_allow_html=True)
    return getattr(where, kind)(label, *args, label_visibility="collapsed", **kw)


def sidebar() -> dict:
    s = load_settings()
    sb = st.sidebar
    p = {}
    head, sw = sb.columns([3, 2], vertical_alignment="center")
    head.markdown("### ⚙️ Your rules")
    p["layman"] = sw.toggle("Layman", value=bool(s["layman"]), key="layman")
    if not p["layman"]:  # hide every hover explanation across the app
        st.markdown("""<style>
[data-tip]:hover::after, [data-tip]:hover::before { display: none !important; }
[data-tip] { cursor: inherit !important; }
.sw-q, .sw-float { display: none !important; }
</style>""", unsafe_allow_html=True)
    p["max_risk"] = field(sb, "number_input", 
        "💸 Max loss per trade (USD)", 1.0, 100_000.0, float(s["max_risk"]), 5.0,
        help="The most you're willing to lose if the stop is hit. The app works out how many shares "
             "to buy so that hitting the stop costs about this much. Shares = max loss ÷ (entry − stop).")
    p["rr"] = field(sb, "number_input", 
        "🎯 Reward : risk (1 : x)", MIN_RR, 10.0, max(float(s["rr"]), MIN_RR), 0.5,
        help="How much you aim to make for every $1 you risk. 1:2 means risk $35 to make $70. "
             "1:2 is the minimum. Higher puts the target further away, so it gets hit less often.")
    p["stop_mode"] = field(sb, "radio", 
        "🛑 Stop based on", ["ATR", "Target %"], index=["ATR", "Target %"].index(s["stop_mode"]),
        horizontal=True,
        help="ATR = the stock's average daily move. 'ATR' puts the stop outside normal daily wiggles, "
             "and the target follows from your reward:risk. 'Target %' lets you pick the rebound "
             "(e.g. 4%) and the stop follows. Too tight a stop gets hit by noise.")
    if p["stop_mode"] == "ATR":
        p["atr_mult"] = field(sb, "number_input", 
            "Stop distance (x ATR)", 0.25, 4.0, float(s["atr_mult"]), 0.25,
            help="1.5 means the stop sits 1.5 normal days' worth of movement below your entry.")
        p["target_pct"] = float(s["target_pct"])
    else:
        p["target_pct"] = field(sb, "number_input", 
            "Rebound target (%)", 1.0, 20.0, float(s["target_pct"]), 0.5,
            help="How far up you expect the bounce to go. With 1:2, a 4% target means a 2% stop.")
        p["atr_mult"] = float(s["atr_mult"])
    p["max_hold"] = field(sb, "number_input", 
        "⏳ Max holding days", 1, 30, int(s["max_hold"]),
        help="If neither the stop nor the target is hit by this day, you exit anyway. "
             "Stops you holding a trade just because it hasn't recovered.")
    with sb.expander("🔎 Scan filters"):
        p["rsi_min"], p["rsi_max"] = field(st, "slider", 
            "🌡️ RSI range", 5, 50, (int(s["rsi_min"]), int(s["rsi_max"])),
            help="RSI (0-100) measures how hard a stock has been sold or bought lately. Under 30 = "
                 "oversold (sellers exhausted). Under about 15-20 often means real bad news, "
                 "a 'falling knife'. Stocks below the lower number still show, under Skipped.")
        p["bb_tol"] = 1 + field(st, "slider", 
            "📏 Lower-band tolerance (%)", 0.0, 5.0, round((float(s["bb_tol"]) - 1) * 100, 1), 0.5,
            help="The lower Bollinger Band marks 'unusually far below the 20-day average'. 0% = price "
                 "must touch or break below the band. 1% = within 1% above it also counts.") / 100
        p["min_rel_vol"] = field(st, "slider", 
            "🔊 Min relative volume (x)", 1.0, 5.0, float(s["min_rel_vol"]), 0.1,
            help="Today's trading volume compared with a normal day. 1.5x = 50% more shares traded "
                 "than usual, a sign of panic or forced selling rather than a quiet drift.")
        p["min_mcap_b"] = field(st, "number_input", 
            "🏢 Min market cap (USD B)", 0.1, 500.0, float(s["min_mcap_b"]), 0.5,
            help="Company size (share price × shares). Bigger companies are less likely to collapse "
                 "for good after a bad day.")
        p["min_price"] = field(st, "number_input", 
            "💲 Min share price", 1.0, 500.0, float(s["min_price"]), 1.0,
            help="Skips very cheap 'penny' stocks, which jump around wildly.")
        p["min_avg_vol"] = field(st, "number_input", 
            "💧 Min avg daily volume (shares)", 100_000, 50_000_000, int(s["min_avg_vol"]), 100_000,
            help="Liquidity. Busy stocks are easy to buy and sell near the price you see.")
        p["earn_buffer"] = field(st, "number_input", 
            "📣 Earnings buffer (days)", 0, 30, int(s["earn_buffer"]),
            help="Skips stocks reporting earnings within this many days. Earnings can make a stock "
                 "jump past your stop overnight.")
    if not HOSTED and sb.button("💾 Save as my defaults", width="stretch"):
        save_settings(p)
        sb.success("Saved.")
    sb.caption("Data: TradingView (unofficial, ~15 min delayed) and Google News. "
               "Research only, not financial advice.")
    return p


# ---------------------------------------------------------------- UI: candidates
def evaluate(row, p: dict, session: str) -> dict:
    t = row["name"]
    ref = ref_price(row, session)
    news = fetch_news(t, str(row["description"]), 6)
    entry = st.session_state.get(f"entry_{t}", ref)
    plan = build_plan(float(entry), float(row["ATR"]), p)
    row = row.copy()
    row["bb_dist_pct"] = 100 * (plan["entry"] / row["BB.lower"] - 1)
    cks = checks(row, plan, news, p)
    failed = [why for _, why, ok in cks if not ok]
    return {"row": row, "ref": ref, "news": news, "plan": plan, "cks": cks, "failed": failed}


def tv_symbol(row) -> str:
    return str(row.get("ticker") or row["name"])


def tv_url(row) -> str:
    return "https://www.tradingview.com/chart/?symbol=" + quote(tv_symbol(row))


def chart_widget(row) -> str:
    """TradingView's free embeddable chart: daily candles with Bollinger Bands and RSI."""
    cfg = {
        "autosize": True, "symbol": tv_symbol(row), "interval": "D",
        "timezone": "America/New_York", "theme": "dark", "style": "1", "locale": "en",
        "backgroundColor": "rgba(18,24,41,1)", "gridColor": "rgba(35,44,69,0.5)",
        "hide_side_toolbar": True, "hide_volume": False, "allow_symbol_change": False,
        "studies": ["STD;Bollinger_Bands", "STD;RSI"], "support_host": "https://www.tradingview.com",
    }
    return f"""<style>html,body{{margin:0;height:100%;background:#121829}}</style>
<div class="tradingview-widget-container" style="height:460px;width:100%">
<div class="tradingview-widget-container__widget" style="height:460px;width:100%"></div>
<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
{json.dumps(cfg)}
</script></div>"""


TIPS = {
    "day": "How much the price moved in the latest trading session. A big red number after bad news is the kind of shock this setup looks for.",
    "rsi": "RSI 0-100: how hard it's been sold lately. Under 30 = oversold. Under 15-20 = possible falling knife.",
    "vol": "Volume vs a normal day. 2x = twice the usual shares traded: a sign of panic selling.",
    "week": "Price change over the last 5 trading days. A deep weekly drop means the selling has been going on for a while, not just one bad day.",
    "entry": "Suggested buy price (latest price, ~15 min delayed). 'max' = above this it's no longer at the lower band, so don't chase.",
    "stop": "If the price falls to here, you sell. This caps the loss at about your max loss. Never move it lower once you're in.",
    "target": "Take-profit price. It sits reward:risk times further above your entry than the stop sits below it (2x for 1:2).",
}


def card_html(ev: dict, p: dict, session: str) -> str:
    row, plan, failed = ev["row"], ev["plan"], ev["failed"]
    esc = html.escape
    rsi = float(row["RSI"])
    rsi_col = "var(--green)" if rsi < 30 else "var(--amber)" if rsi < 40 else "var(--muted)"
    vol = float(row["relative_volume_10d_calc"])
    week = row["Perf.W"]
    week_txt = "n/a" if pd.isna(week) else f"{week:+.1f}%"
    chip = ('<span class="sw-chip no">⛔ Skip</span>' if failed
            else '<span class="sw-chip go">✨ Worth a look</span>')
    cut = 100 / (1 + p["rr"])
    bb_line = row["BB.lower"] * p["bb_tol"]

    tags = []
    if row["source"] == "premarket gap":
        pm = row["premarket_change"]
        tags.append(f"🌙 Overnight gap {pm:+.1f}%, indicators from last close")
    atr_pct = 100 * row["ATR"] / row["close"]
    if plan["stop_atr"] < 0.75:
        tags.append(f"🌊 Stop is inside daily noise ({atr_pct:.1f}%/day)")
    if plan["target_pct"] > 5:
        tags.append(f"🚀 Target +{plan['target_pct']:.1f}% is a big ask")
    last = days_from_now(row["earnings_release_date"])
    if last is not None and -4 <= last <= 0:
        tags.append("📣 Drop follows earnings, read the report")
    if row["close"] < row["SMA200"]:
        tags.append("📉 Below 200-day average")

    why = f'<div class="sw-why">⛔ {esc("; ".join(failed))}</div>' if failed else ""
    tag_html = ('<div class="sw-tags">' + "".join(f'<span class="sw-tag">{esc(t)}</span>' for t in tags)
                + "</div>") if tags else ""
    return f"""
<div class="sw sw-card {'skip' if failed else ''}">
  <div class="sw-top">
    <div><div class="sw-tkr"><a href="{tv_url(row)}" target="_blank">{esc(row['name'])}<span>↗</span></a></div>
         <div class="sw-co">{esc(str(row['description']))}</div>
         <div class="sw-co">🏭 {esc(str(row['sector']))} › {esc(str(row['industry']))}</div></div>
    {chip}
  </div>
  <div class="sw-mini">
    <div data-tip="{TIPS['day']}"><div class="k">Day</div><div class="v red">{row['change']:+.1f}%</div></div>
    <div data-tip="{TIPS['rsi']}"><div class="k">RSI</div><div class="v">{rsi:.0f}</div>
         <div class="sw-meter"><span style="width:{min(rsi, 100):.0f}%;background:{rsi_col}"></span></div></div>
    <div data-tip="{TIPS['vol']}"><div class="k">Vol</div><div class="v {'green' if vol >= p['min_rel_vol'] else ''}">{vol:.1f}x</div></div>
    <div data-tip="{TIPS['week']}"><div class="k">Week</div><div class="v {'red' if not pd.isna(week) and week < 0 else ''}">{week_txt}</div></div>
  </div>
  <div class="sw-plan">
    <div class="e" data-tip="{TIPS['entry']}"><div class="k">🎯 ENTRY</div><div class="v">{money(plan['entry'])}</div>
         <div class="d" style="color:var(--muted)">max {money(bb_line)}</div></div>
    <div class="s" data-tip="{TIPS['stop']}"><div class="k">🛑 STOP</div><div class="v">{money(plan['stop'])}</div>
         <div class="d red">-{plan['stop_pct']:.1f}%</div></div>
    <div class="t" data-tip="{TIPS['target']}"><div class="k">🏁 TARGET</div><div class="v">{money(plan['target'])}</div>
         <div class="d green">+{plan['target_pct']:.1f}%</div></div>
  </div>
  <div class="sw-ladder" style="--cut:{cut:.1f}%"><i></i></div>
  <div class="sw-foot">
    <span>🧮 <b>{plan['shares']}</b> shares · <b>{money(plan['position'])}</b></span>
    <span>💸 risk <b>{money(plan['risk'])}</b> · 1:{p['rr']:g}</span>
    <span>⏳ exit by day <b>{p['max_hold']}</b></span>
  </div>
  {why}{tag_html}
</div>"""


def render_candidate(ev: dict, p: dict, session: str):
    row = ev["row"]
    t = row["name"]
    st.markdown(card_html(ev, p, session), unsafe_allow_html=True)
    with st.expander(f"📈 Chart  ·  📰 why did {t} drop?  ·  ✅ checks"):
        if st.toggle("📈 Show live chart (Bollinger Bands + RSI)", key=f"chart_{t}"):
            components.html(chart_widget(row), height=465)
        st.number_input("✏️ Your entry price (plan recalculates)", 0.01, 100_000.0,
                        float(round(ev["ref"], 2)), 0.01, key=f"entry_{t}")
        news_html = "".join(
            f'<div class="sw-news">{ {"severe": "🔴 ", "caution": "🟡 ", "error": "⚠️ "}.get(n["level"], "🗞️ ")}'
            + (f'<a href="{html.escape(n["link"])}" target="_blank">{html.escape(n["title"])}</a>' if n["link"]
               else html.escape(n["title"]))
            + f'<div class="src">{html.escape(n["source"])} · {html.escape(n["published"])}</div></div>'
            for n in ev["news"])
        st.markdown(news_html, unsafe_allow_html=True)
        st.caption("🔴 🟡 are keyword flags, not a judgment. Is this emotion, or the business changing?")
        st.markdown("".join(f'<div class="sw-check">{"✅" if ok else "❌"} {html.escape(label)}</div>'
                            for label, _, ok in ev["cks"]), unsafe_allow_html=True)


# Coloured for a buy-the-panic trader: fear = opportunity (green), greed = be careful (red).
FG_COLORS = {"extreme fear": "#2BD9A0", "fear": "#8EE6A8", "neutral": "#FFC94D",
             "greed": "#FF5C7A", "extreme greed": "#B3243B"}
FG_TIP = ("CNN Fear & Greed Index: the mood of the US stock market from 7 signals (momentum, volatility, "
          "safe-haven demand and more). 0 = extreme fear (people panic-selling), 100 = extreme greed. "
          "Green (fear) = panic selling, when buy-the-panic setups tend to appear. It is not a buy signal by itself: "
          "fear can last weeks, so each trade still has to pass your rules. Red (greed) = market is "
          "euphoric, be extra careful. Click to open CNN.")


def fear_greed_html() -> str:
    fg = fetch_fear_greed()
    if fg is None:
        return (f'<a class="sw-fg" href="{FG_PAGE}" target="_blank" data-tip="{FG_TIP}" style="display:block">'
                '<div class="k">😨 Fear &amp; Greed</div><div class="r" style="color:var(--muted)">'
                'Unavailable right now</div><div class="m"><span>Source: CNN ↗</span></div></a>')
    c = FG_COLORS.get(fg["rating"], "#FFC94D")
    prev = f"prev {fg['prev']:.0f}" if fg["prev"] is not None else ""
    week = f"1w ago {fg['week']:.0f}" if fg["week"] is not None else ""
    return f"""<div class="sw-fg" data-tip="{FG_TIP}">
  <div class="k">😨 Fear &amp; Greed index</div>
  <div class="n" style="color:{c}">{fg['score']:.0f}</div>
  <div class="r" style="color:{c}">{html.escape(fg['rating'])}</div>
  <div class="bar"><i style="left:{min(max(fg['score'], 0), 100):.0f}%"></i></div>
  <div class="m"><span>{prev} · {week}</span><a href="{FG_PAGE}" target="_blank">CNN ↗</a></div>
</div>"""


def hero(session: str):
    now_et = datetime.now(ET_TZ)
    label, target = next_session_event()
    cls, emoji = {"open": ("live", "🟢"), "premarket": ("pre", "🌅"),
                  "after hours": ("pre", "🌙")}.get(session, ("off", "🔴"))
    st.markdown(f"""
<div class="sw sw-hero"><div>
  <h1>📉 SkimTrade</h1>
  <div class="sub">Find oversold rebounds after emotional sell-offs. Buy the panic, not the problem.</div>
  <div class="sw-pills">
    <span class="sw-pill {cls}">{emoji} Market {session}</span>
    <span class="sw-pill cd">⏳ {label} in <b class="sw-cd" data-until="{int(target.timestamp() * 1000)}">{countdown(target)}</b></span>
    <span class="sw-pill">🗽 New York <span class="sw-clock" data-tz="America/New_York">{now_et:%a %H:%M}</span></span>
    <span class="sw-pill">🇸🇬 Singapore <span class="sw-clock" data-tz="Asia/Singapore">{now_et.astimezone(SG_TZ):%a %H:%M}</span></span>
    <span class="sw-pill">⏱️ ~15 min delayed</span>
    <span class="sw-pill">🎯 Lower band · RSI · volume · news</span>
  </div></div>
  {fear_greed_html()}
</div>""", unsafe_allow_html=True)


def stats_html(items: list) -> str:
    return ('<div class="sw sw-stats">' + "".join(
        '<div class="sw-stat"' + (f' data-tip="{html.escape(tip)}"' if tip else "") +
        f'><div class="n">{n}</div><div class="l">{l}</div></div>'
        for n, l, tip in items) + "</div>")


def main():
    st.set_page_config(page_title="SkimTrade", page_icon="📉", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    components.html(TIP_JS, height=0)
    p = sidebar()
    session = market_session()
    hero(session)

    c1, c2 = st.columns([2, 1])
    with c1:
        if session.startswith("closed") or session == "after hours":
            st.caption("😴 Market is closed, so this is the last close. Best times to scan: premarket "
                       "for overnight shocks, and right after the open.")
        else:
            st.caption("⚡ Market is live. Prices refresh each time you scan.")
    run = c2.button("🔍 Scan the market", type="primary", width="stretch")
    if run:
        with st.spinner("Scanning the market for panic sells..."):
            try:
                st.session_state["raw"] = run_scan(p, session)
                st.session_state["ran_at"] = datetime.now(ET_TZ).strftime("%a %H:%M")
                st.session_state["session"] = session
            except Exception as e:
                st.session_state.pop("raw", None)
                st.error(f"Scan failed: {type(e).__name__}: {e}")

    if "raw" not in st.session_state:
        st.markdown(stats_html([("👆", "Press scan to start", ""),
                                ("🎯", "1:" + f"{p['rr']:g}" + " reward:risk", ""),
                                ("💸", f"${p['max_risk']:g} max loss", ""),
                                ("⏳", f"{p['max_hold']} day max hold", "")]),
                    unsafe_allow_html=True)
        return

    raw = st.session_state["raw"].copy()
    sess = st.session_state["session"]
    if raw.empty:
        st.success("😌 Nothing is oversold on heavy volume. No setup means no trade.")
        return
    raw["ref"] = [ref_price(r, sess) for _, r in raw.iterrows()]
    near = raw[(raw["ref"] <= raw["BB.lower"] * p["bb_tol"]) | (raw["source"] == "premarket gap")]
    near = near.sort_values(["RSI", "relative_volume_10d_calc"], ascending=[True, False]).head(16)
    evs = [evaluate(row, p, sess) for _, row in near.iterrows()]
    good = [e for e in evs if not e["failed"]]
    skip = [e for e in evs if e["failed"]]

    st.markdown(stats_html([
        (int((raw["source"] == "sell-off").sum()), "📉 Oversold on volume",
         f"RSI under {p['rsi_max']} with at least {p['min_rel_vol']}x normal volume."),
        (len(near), "🎯 At the lower band", "Of those, how many are at or below the lower Bollinger Band."),
        (len(good), "✨ Worth a look", "Passed every automatic check. Still read the news before deciding."),
        (int((raw["source"] == "premarket gap").sum()), "🌙 Premarket gaps",
         f"Stocks down more than {abs(PREMARKET_GAP):g}% before the open (overnight shocks). Only during premarket."),
    ]), unsafe_allow_html=True)
    st.caption(f"Last scan {st.session_state['ran_at']} New York time. "
               "Passing every check only means nothing ruled it out. The call is yours.")

    if not evs:
        st.warning("Some stocks are oversold, but none are at the lower Bollinger Band. No trade today. 🧘")
        return
    if good:
        st.markdown('<div class="sw sw-section">✨ Worth a look</div>', unsafe_allow_html=True)
        cols = st.columns(2, gap="medium")
        for i, e in enumerate(good):
            with cols[i % 2]:
                render_candidate(e, p, sess)
    if skip:
        with st.expander(f"⛔ Skipped ({len(skip)}): they failed a rule"):
            cols = st.columns(2, gap="medium")
            for i, e in enumerate(skip):
                with cols[i % 2]:
                    render_candidate(e, p, sess)


if __name__ == "__main__":
    main()
