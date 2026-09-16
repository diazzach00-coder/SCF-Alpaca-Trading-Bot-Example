"""Pull 1-minute bars from Alpaca into data/<SYMBOL>_1min.parquet (free plan is enough).

    python pull_data.py                       # QQQ SPY IWM, 2016-01-04 -> now
    python pull_data.py QQQ --start 2020-01-01

Uses the consolidated (SIP) feed, split-adjusted. The free plan serves SIP history as long
as the request ends at least 15 minutes in the past, so we end 20 minutes ago. Ten years
of one ETF is ~1 million bars and takes a few minutes; existing files are skipped.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from bot import DATA_HOST, load_secrets

DATA = Path(__file__).resolve().parent / "data"


def fetch(symbol: str, start: str, end: str, headers: dict) -> list[dict]:
    rows, token = [], None
    while True:
        r = requests.get(f"{DATA_HOST}/v2/stocks/bars", headers=headers, timeout=60,
                         params={"symbols": symbol, "timeframe": "1Min", "start": start, "end": end, "feed": "sip",
                                 "adjustment": "split", "limit": 10000, "sort": "asc", "page_token": token})
        if r.status_code == 429:
            time.sleep(6)
            continue
        if r.status_code != 200:
            raise SystemExit(f"{symbol}: HTTP {r.status_code} {r.text[:200]}")
        body = r.json()
        rows.extend((body.get("bars") or {}).get(symbol, []))
        token = body.get("next_page_token")
        if not token:
            return rows
        time.sleep(0.32)                                          # stay under 200 requests/min


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["QQQ", "SPY", "IWM"])
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--force", action="store_true", help="re-pull even if the file exists")
    a = ap.parse_args()
    headers = load_secrets()
    DATA.mkdir(exist_ok=True)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    for sym in [s.upper() for s in a.symbols]:
        out = DATA / f"{sym}_1min.parquet"
        if out.exists() and not a.force:
            print(f"{sym}: {out.name} exists, skipping (use --force to re-pull)")
            continue
        frames, t0 = [], time.time()
        for year in range(int(a.start[:4]), end.year + 1):
            s = a.start if year == int(a.start[:4]) else f"{year}-01-01"
            e = end.strftime("%Y-%m-%dT%H:%M:%SZ") if year == end.year else f"{year + 1}-01-01"
            rows = fetch(sym, s, e, headers)
            if rows:
                frames.append(pd.DataFrame(rows))
            print(f"  {sym} {year}: {len(rows):>8,} bars  ({time.time() - t0:4.0f}s)", flush=True)
        if not frames:
            print(f"{sym}: no bars returned")
            continue
        df = pd.concat(frames, ignore_index=True).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
        df[["t", "open", "high", "low", "close", "volume"]].to_parquet(out, index=False)
        print(f"-> {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
