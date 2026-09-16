"""A day-trading bot for Alpaca PAPER accounts. One file. Read it top to bottom.

    python bot.py                    # dry run: decides and logs, sends nothing
    python bot.py --live-paper       # sends PAPER orders (the trading host must contain "paper-api")
    python bot.py --lev-cap 4        # bigger swings (the paper's leverage cap; the backtest default is 1x)
    python bot.py --help             # every knob

The rule (Zarattini, Barbon & Aziz 2024, "Beat the Market"):
  09:30  build today's band from the last 14 sessions: max/min(open, yesterday's close)
         x (1 +/- the average |move from the open| at each minute of the day)
  10:00 -> 15:30, every CHECK_EVERY minutes: price above the band -> long,
         below -> short, inside -> flat. The band is the trailing stop.
  every EXIT_EVERY minutes in between: an open trade is closed if price is back inside
  15:55  flat. Nothing is held overnight.
  size   = NOTIONAL_FRACTION of equity x min(VOL_TARGET / 14-day daily vol, LEV_CAP)

Safety: there is no live-money code path. Orders go only to a host containing "paper-api",
and only with --live-paper. Everything the bot sees and decides is logged to logs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

# ============================================================================ settings
SYMBOLS = ["QQQ", "SPY"]          # the backtest says QQQ works, SPY half-works, IWM loses
CHECK_EVERY = 30                  # minutes between decisions (faster costs edge - see backtest.py --ladder)
EXIT_EVERY = 5                    # minutes between stop-only checks while in a trade (0 = none)
FIRST_DECISION = (10, 0)          # ET
EXIT_TIME = (15, 55)              # ET
LOOKBACK = 14                     # sessions used for the band and the vol estimate
VOL_TARGET = 0.02                 # daily
LEV_CAP = 1.0                     # per symbol; 4.0 = the paper's cap = ~4x the swings AND drawdowns
NOTIONAL_FRACTION = 0.5           # of equity per symbol at leverage 1
LONG_ONLY = False

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
DATA_HOST = "https://data.alpaca.markets"
RTH_MINUTES = 390


# ============================================================================ secrets
def load_secrets(path: Path = HERE / "secrets.env") -> dict:
    """Read KEY=VALUE lines. Returns the auth headers Alpaca wants. Never print these."""
    if not path.exists():
        raise SystemExit(f"missing {path}: copy secrets.env.example to secrets.env and paste your PAPER keys")
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    if not env.get("APCA_API_KEY_ID") or not env.get("APCA_API_SECRET_KEY"):
        raise SystemExit("secrets.env needs APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": env["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": env["APCA_API_SECRET_KEY"],
            "_trading_host": env.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")}


# ============================================================================ the rule (pure functions)
def sigma_profile(history: list[tuple[np.ndarray, float, np.ndarray]], lookback: int) -> np.ndarray:
    """history: per past session (minute_of_day array, that day's open, closes).
    Returns 390 values: the average |close/open - 1| at each minute over the last `lookback` days."""
    rows = []
    for m, o, c in history[-lookback:]:
        v = np.full(RTH_MINUTES, np.nan)
        idx = np.asarray(m) - 570
        ok = (idx >= 0) & (idx < RTH_MINUTES)
        v[idx[ok]] = np.abs(np.asarray(c, float)[ok] / float(o) - 1.0)
        rows.append(v)
    if not rows:
        return np.full(RTH_MINUTES, np.nan)
    with np.errstate(all="ignore"):
        prof = np.nanmean(np.vstack(rows), axis=0)
    last = np.nan                                     # forward-fill minutes with no bar
    for i in range(RTH_MINUTES):
        if np.isfinite(prof[i]):
            last = prof[i]
        elif np.isfinite(last):
            prof[i] = last
    return prof


def daily_vol(closes: list[float], lookback: int) -> float | None:
    c = np.asarray([x for x in closes if x and x > 0][-(lookback + 1):], float)
    if c.size < 6:
        return None
    r = np.diff(np.log(c))
    return float(np.std(r, ddof=1))


def bands(sigma_t: float, open_px: float, prev_close: float) -> tuple[float, float]:
    hi, lo = max(open_px, prev_close), min(open_px, prev_close)
    return hi * (1.0 + sigma_t), lo * (1.0 - sigma_t)


def target_from_price(price: float, ub: float, lb: float, long_only: bool = False) -> int:
    if price > ub:
        return 1
    if price < lb:
        return 0 if long_only else -1
    return 0


def shares_for(equity: float, price: float, vol_d: float | None, lev_cap: float) -> tuple[int, float]:
    if equity <= 0 or price <= 0:
        return 0, 0.0
    lev = min(VOL_TARGET / vol_d, lev_cap) if vol_d and vol_d > 0 else 1.0
    return int(math.floor(equity * NOTIONAL_FRACTION * lev / price)), lev


def decision_slot(now: dt.datetime, check_every: int) -> int | None:
    """Which decision window `now` is in (0 = 10:00-10:29, 1 = 10:30-10:59, ...). The loop acts on the
    first iteration inside a new window, so a slow iteration can delay a decision but never skip it."""
    m = now.hour * 60 + now.minute
    first = FIRST_DECISION[0] * 60 + FIRST_DECISION[1]
    if m < first or (now.hour, now.minute) >= EXIT_TIME:
        return None
    return (m - first) // check_every


def exit_slot(now: dt.datetime, exit_every: int) -> int | None:
    if not exit_every:
        return None
    m = now.hour * 60 + now.minute
    first = FIRST_DECISION[0] * 60 + FIRST_DECISION[1]
    if m < first or (now.hour, now.minute) >= EXIT_TIME:
        return None
    return (m - first) // exit_every


# ============================================================================ Alpaca I/O
class Alpaca:
    def __init__(self, headers: dict):
        self.h = {k: v for k, v in headers.items() if not k.startswith("_")}
        self.trading_host = headers["_trading_host"]

    def clock(self) -> dict:
        return requests.get(f"{self.trading_host}/v2/clock", headers=self.h, timeout=30).json()

    def equity(self) -> float:
        return float(requests.get(f"{self.trading_host}/v2/account", headers=self.h, timeout=30).json()["equity"])

    def position(self, sym: str) -> int:
        r = requests.get(f"{self.trading_host}/v2/positions/{sym}", headers=self.h, timeout=30)
        if r.status_code != 200:
            return 0
        q = int(float(r.json().get("qty", 0)))
        return -q if r.json().get("side") == "short" else q

    def bars(self, sym: str, start: dt.datetime, end: dt.datetime, feed: str) -> list[dict]:
        rows, token = [], None
        while True:
            r = requests.get(f"{DATA_HOST}/v2/stocks/{sym}/bars", headers=self.h, timeout=60,
                             params={"timeframe": "1Min", "feed": feed, "adjustment": "raw", "limit": 10000,
                                     "start": start.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "end": end.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "page_token": token})
            if r.status_code != 200:
                raise RuntimeError(f"bars {sym} HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            rows.extend(body.get("bars") or [])
            token = body.get("next_page_token")
            if not token:
                return rows

    def latest_price(self, sym: str) -> float | None:
        r = requests.get(f"{DATA_HOST}/v2/stocks/{sym}/trades/latest", headers=self.h, params={"feed": "iex"}, timeout=30)
        if r.status_code != 200:
            return None
        return float(r.json().get("trade", {}).get("p") or 0) or None

    def submit(self, payload: dict) -> dict:
        r = requests.post(f"{self.trading_host}/v2/orders", headers=self.h, json=payload, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"order rejected HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


# ============================================================================ the bot
class Bot:
    def __init__(self, api: Alpaca, symbols: list[str], *, live_paper: bool, check_every: int, exit_every: int,
                 lev_cap: float, long_only: bool):
        self.api, self.symbols = api, symbols
        self.live_paper, self.check_every, self.exit_every, self.lev_cap, self.long_only = live_paper, check_every, exit_every, lev_cap, long_only
        if live_paper and "paper-api" not in api.trading_host:
            raise SystemExit("refusing: --live-paper needs a trading host that contains 'paper-api'. There is no live path.")
        self.held: dict[str, int] = {}
        self.today_open: dict[str, float] = {}
        self.prev_close: dict[str, float] = {}
        self.vol_d: dict[str, float | None] = {}
        self.sigma: dict[str, np.ndarray] = {}
        self.equity = 0.0
        (HERE / "logs").mkdir(exist_ok=True)
        self.logfile = HERE / "logs" / f"bot_{dt.datetime.now(ET):%Y-%m-%d}.log"

    def log(self, msg: str, **fields) -> None:
        line = {"ts": dt.datetime.now(ET).isoformat(timespec="seconds"), "msg": msg, **fields}
        with self.logfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"[{line['ts'][11:19]}] {msg} {fields if fields else ''}", flush=True)

    # ---- data
    def load_history(self, sym: str, today: dt.date) -> None:
        """The last ~14 sessions of consolidated 1-min bars (free plan: request must end >= 15 min ago)."""
        end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
        rows = self.api.bars(sym, end - dt.timedelta(days=int(LOOKBACK * 1.7) + 8), end, feed="sip")
        by_day: dict[dt.date, list] = {}
        for b in rows:
            ts = dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            m = ts.hour * 60 + ts.minute
            if 570 <= m < 960 and ts.date() < today:
                by_day.setdefault(ts.date(), []).append((m, float(b["o"]), float(b["c"])))
        days = []
        for d in sorted(by_day):
            bars = sorted(by_day[d])
            if len(bars) >= 300:
                days.append((np.array([x[0] for x in bars]), bars[0][1], np.array([x[2] for x in bars])))
        closes = [float(c[-1]) for _, _, c in days]
        self.prev_close[sym] = closes[-1]
        self.vol_d[sym] = daily_vol(closes, LOOKBACK)
        self.sigma[sym] = sigma_profile(days, LOOKBACK)
        self.log("history loaded", symbol=sym, sessions=len(days), prev_close=closes[-1], vol_daily=self.vol_d[sym])

    def fetch_today_open(self, sym: str, now: dt.datetime) -> float | None:
        rows = self.api.bars(sym, now.replace(hour=9, minute=30, second=0, microsecond=0), now, feed="iex")   # IEX = real time, free
        return float(rows[0]["o"]) if rows else None

    def mark(self, sym: str, now: dt.datetime) -> dict | None:
        px = self.api.latest_price(sym)
        if px is None or sym not in self.today_open:
            return None
        i = min(max(now.hour * 60 + now.minute - 570, 0), RTH_MINUTES - 1)
        sg = float(self.sigma[sym][i])
        if not np.isfinite(sg):
            return None
        ub, lb = bands(sg, self.today_open[sym], self.prev_close[sym])
        return {"price": px, "ub": round(ub, 3), "lb": round(lb, 3), "held": self.held.get(sym, 0)}

    # ---- orders (guarded)
    def set_position(self, sym: str, target: int) -> None:
        delta = target - self.held.get(sym, 0)
        if delta == 0:
            return
        payload = {"symbol": sym, "qty": str(abs(delta)), "side": "buy" if delta > 0 else "sell",
                   "type": "market", "time_in_force": "day"}
        if not self.live_paper:
            self.log(f"DRY RUN - would submit {payload['side']} {abs(delta)} {sym} -> target {target}")
            return
        order = self.api.submit(payload)
        self.log(f"submitted {payload['side']} {abs(delta)} {sym} -> target {target}", order_id=order.get("id"), status=order.get("status"))
        self.held[sym] = target

    # ---- the day
    def run_day(self) -> None:
        now = dt.datetime.now(ET)
        clk = self.api.clock()
        next_open = dt.datetime.fromisoformat(clk["next_open"].replace("Z", "+00:00")).astimezone(ET)
        self.equity = self.api.equity()
        self.log("start", is_open=clk.get("is_open"), equity=self.equity, live_paper=self.live_paper, symbols=self.symbols,
                 check_every=self.check_every, exit_every=self.exit_every, lev_cap=self.lev_cap, long_only=self.long_only)
        if not (clk.get("is_open") or next_open.date() == now.date()):
            self.log("not a trading day (or after the close) - nothing to do")
            return
        keep_awake(True)
        try:
            open_at = now.replace(hour=9, minute=30, second=30, microsecond=0)
            if now < open_at:
                self.log("waiting for the open", seconds=int((open_at - now).total_seconds()))
                while dt.datetime.now(ET) < open_at:
                    time.sleep(min(30.0, max(1.0, (open_at - dt.datetime.now(ET)).total_seconds())))
            today = dt.datetime.now(ET).date()
            for sym in self.symbols:
                self.load_history(sym, today)
                if self.live_paper:
                    self.held[sym] = self.api.position(sym)          # resume if restarted mid-day
            for _ in range(20):                                     # the open print can lag a minute on IEX
                for sym in self.symbols:
                    if sym not in self.today_open:
                        o = self.fetch_today_open(sym, dt.datetime.now(ET))
                        if o:
                            self.today_open[sym] = o
                            self.log("today's open", symbol=sym, open=o, prev_close=self.prev_close[sym])
                if all(s in self.today_open for s in self.symbols):
                    break
                time.sleep(15)

            acted_slot, acted_exit = None, None
            while True:
                now = dt.datetime.now(ET)
                if (now.hour, now.minute) >= EXIT_TIME:
                    for sym in self.symbols:
                        self.set_position(sym, 0)
                    self.log("flat at exit time", book=self.held)
                    break
                slot, xslot = decision_slot(now, self.check_every), exit_slot(now, self.exit_every)
                decide = slot is not None and slot != acted_slot
                exit_check = (not decide) and xslot is not None and xslot != acted_exit
                for sym in self.symbols:
                    mk = self.mark(sym, now)
                    if mk is None:
                        continue
                    if decide:
                        tgt = target_from_price(mk["price"], mk["ub"], mk["lb"], self.long_only)
                        sh, lev = shares_for(self.equity, mk["price"], self.vol_d.get(sym), self.lev_cap)
                        self.log("decision", symbol=sym, call={1: "long", -1: "short", 0: "flat"}[tgt], target_shares=tgt * sh, lev=round(lev, 2), **mk)
                        self.set_position(sym, tgt * sh)
                    elif exit_check and mk["held"] != 0:
                        tgt = target_from_price(mk["price"], mk["ub"], mk["lb"], self.long_only)
                        if tgt != (1 if mk["held"] > 0 else -1):
                            self.log("stop check: back inside the band -> flat", symbol=sym, **mk)
                            self.set_position(sym, 0)
                        else:
                            self.log("stop check: still through the band -> hold", symbol=sym, **mk)
                    else:
                        self.log("mark", symbol=sym, **mk)
                if decide:
                    acted_slot, acted_exit = slot, xslot
                elif exit_check:
                    acted_exit = xslot
                time.sleep(max(1.0, 61.0 - dt.datetime.now(ET).second))   # wake just after each minute boundary
        finally:
            keep_awake(False)


def keep_awake(on: bool) -> None:
    """Tell Windows the process is busy so the machine does not idle-sleep. A closed lid still sleeps it."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 if on else 0x80000000)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-paper", action="store_true", help="send PAPER orders (default: dry run)")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--check-every", type=int, default=CHECK_EVERY, help="minutes between decisions")
    ap.add_argument("--exit-every", type=int, default=EXIT_EVERY, help="minutes between stop-only checks (0 = none)")
    ap.add_argument("--lev-cap", type=float, default=LEV_CAP, help="leverage cap per symbol (1 = backtest, 4 = the paper)")
    ap.add_argument("--long-only", action="store_true", default=LONG_ONLY)
    a = ap.parse_args()
    bot = Bot(Alpaca(load_secrets()), [s.upper() for s in a.symbols], live_paper=a.live_paper, check_every=a.check_every,
              exit_every=a.exit_every, lev_cap=a.lev_cap, long_only=a.long_only)
    bot.run_day()


if __name__ == "__main__":
    main()
