"""Backtest the bot's rule on 1-minute bars, the honest way.

    python backtest.py                              # QQQ SPY IWM: decide every 30 min, stop check every 5, 1 bp
    python backtest.py --symbols QQQ --check-every 15
    python backtest.py --ladder                     # decision clocks 5/10/15/20/30/60 for each symbol
    python backtest.py --cost-bp 2 --lev-cap 4

Needs data/<SYMBOL>_1min.parquet from pull_data.py.

The rule (same code path as bot.py, see strategy_day):
  band  = max/min(today's open, yesterday's close) x (1 +/- sigma[minute]),
          sigma[minute] = average |close/open - 1| at that minute over the previous 14 days
  every CHECK_EVERY minutes: price above band -> long, below -> short, inside -> flat
  every EXIT_EVERY minutes in between: an open trade is closed if price is back inside
  flat at the close; sizing = VOL_TARGET / 14-day daily vol, capped at LEV_CAP

What makes it honest:
  * point in time: the band for day t uses days t-14..t-1 only; the signal is the last bar
    BEFORE the decision minute and the fill is the NEXT bar's open
  * costs charged on every round trip (COST_BP)
  * the Sharpe is deflated for the number of variants you ran (Bailey & Lopez de Prado 2014)
  * the sample is split 2016-19 / 2020-26 so a one-regime edge shows up as one
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats as sps

ET = ZoneInfo("America/New_York")
DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"
TRADING_DAYS = 252
RTH_MINUTES = 390          # 09:30 -> 16:00


# ----------------------------------------------------------------------------- data
def load_days(symbol: str) -> list[tuple]:
    """One tuple per session: (date, minute_of_day array, open array, close array)."""
    f = DATA / f"{symbol}_1min.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f} - run:  python pull_data.py {symbol}")
    b = pd.read_parquet(f)
    et = b["t"].dt.tz_convert(ET)
    minute = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    keep = (minute >= 570) & (minute < 960)                       # regular hours only
    b = b.loc[keep].copy()
    b["m"] = minute[keep]
    b["date"] = et.loc[b.index].dt.date
    days = []
    for d, g in b.sort_values("t").groupby("date", sort=True):
        if len(g) < 350:                                          # skip half days
            continue
        days.append((d, g["m"].to_numpy(), g["open"].to_numpy(float), g["close"].to_numpy(float)))
    return days


# ------------------------------------------------------------------------- strategy
def sigma_table(days: list[tuple], lookback: int) -> np.ndarray:
    """sigma[day, minute] = mean |close/open - 1| at that minute over the PREVIOUS `lookback` days."""
    mv = np.full((len(days), RTH_MINUTES), np.nan)
    for k, (_, m, o, c) in enumerate(days):
        mv[k, m - 570] = np.abs(c / o[0] - 1.0)
    return pd.DataFrame(mv).rolling(lookback, min_periods=10).mean().shift(1).to_numpy()


def strategy_day(m, o, c, sigma_row, prev_close, lev, *, check_every, exit_every, cost, long_only=False):
    """Simulate one day. Returns (return on equity, number of trades)."""
    hi_ref, lo_ref = max(o[0], prev_close), min(o[0], prev_close)
    step = math.gcd(check_every, exit_every) if exit_every else check_every
    pos, entry, day_ret, trades = 0, 0.0, 0.0, 0
    for t in range(570 + step, 960, step):
        is_decision = (t - 570) % check_every == 0
        if not is_decision and pos == 0:
            continue                                              # stop checks only matter in a trade
        s = int(np.searchsorted(m, t, side="left")) - 1           # last bar strictly before t = the signal
        if s < 0 or s + 1 >= len(m):
            break
        sg = sigma_row[m[s] - 570]
        if not np.isfinite(sg):
            continue
        p = c[s]
        target = 1 if p > hi_ref * (1 + sg) else (-1 if p < lo_ref * (1 - sg) else 0)
        if long_only and target < 0:
            target = 0
        if not is_decision and target != pos:
            target = 0                                            # between decisions: exit only, never reverse
        if target != pos:
            fill = float(o[s + 1])                                # next bar's open: no peeking
            if pos != 0:
                day_ret += pos * lev * (fill / entry - 1.0) - lev * cost
            if target != 0:
                entry, trades = fill, trades + 1
            pos = target
    if pos != 0:                                                  # flat at the close
        day_ret += pos * lev * (float(c[-1]) / entry - 1.0) - lev * cost
    return day_ret, trades


def run_strategy(days, *, check_every=30, exit_every=5, lookback=14, vol_target=0.02, lev_cap=1.0,
                 cost=0.0001, long_only=False):
    sigma = sigma_table(days, lookback)
    closes = np.array([c[-1] for _, m, o, c in days])
    r_cc = np.diff(np.log(closes), prepend=np.nan)
    vol = pd.Series(r_cc).rolling(lookback, min_periods=10).std().shift(1).to_numpy()   # through yesterday
    rets, ntr = np.zeros(len(days)), np.zeros(len(days), int)
    for k, (_, m, o, c) in enumerate(days):
        if k == 0 or not np.isfinite(vol[k]) or vol[k] <= 0:
            continue
        lev = min(vol_target / vol[k], lev_cap)
        rets[k], ntr[k] = strategy_day(m, o, c, sigma[k], closes[k - 1], lev, check_every=check_every,
                                       exit_every=exit_every, cost=cost, long_only=long_only)
    return rets, ntr


# ---------------------------------------------------------------------- statistics
def psr(sr: float, n: int, skew: float, kurt: float, sr_b: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true Sharpe > sr_b)."""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if v <= 0 or n <= 1:
        return float("nan")
    return float(sps.norm.cdf((sr - sr_b) * math.sqrt(n - 1.0) / math.sqrt(v)))


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe ratio [Bailey & Lopez de Prado 2014]: PSR against the Sharpe that pure luck
    would produce as the best of `n_trials` tries."""
    if n_trials <= 1 or var_sr <= 0:
        return psr(sr, n, skew, kurt, 0.0)
    gamma = 0.5772156649015329
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * sps.norm.ppf(1 - 1 / n_trials) + gamma * sps.norm.ppf(1 - 1 / (n_trials * math.e)))
    return psr(sr, n, skew, kurt, sr0)


def stats(r: np.ndarray, dates: list) -> dict:
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd if sd > 0 else float("nan")
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    pre = np.array([d < pd.Timestamp("2020-01-01").date() for d in dates])

    def ann_sr(x):
        return float(x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS)) if x.size > 2 and x.std(ddof=1) > 0 else float("nan")

    return {"n_days": int(r.size), "ann_return": float(mu * TRADING_DAYS), "ann_vol": float(sd * math.sqrt(TRADING_DAYS)),
            "sharpe": float(sr * math.sqrt(TRADING_DAYS)), "t_stat": float(mu / (sd / math.sqrt(r.size))) if sd > 0 else float("nan"),
            "max_drawdown": dd, "sharpe_2016_2019": ann_sr(r[pre]), "sharpe_2020_2026": ann_sr(r[~pre]),
            "sr_daily": float(sr), "skew": float(sps.skew(r)), "kurt": float(sps.kurtosis(r, fisher=False))}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY", "IWM"])
    ap.add_argument("--check-every", type=int, default=30, help="minutes between decisions (default 30)")
    ap.add_argument("--exit-every", type=int, default=5, help="minutes between stop-only checks; 0 = none (default 5)")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--vol-target", type=float, default=0.02, help="daily vol target (default 2%%)")
    ap.add_argument("--lev-cap", type=float, default=1.0, help="leverage cap (default 1x; the paper used 4x)")
    ap.add_argument("--cost-bp", type=float, default=1.0, help="cost per round trip in basis points (default 1)")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--ladder", action="store_true", help="run decision clocks 5/10/15/20/30/60 instead of one")
    ap.add_argument("--trials", type=int, default=0, help="variants tried so far, for the deflated Sharpe (default: the ones run now)")
    a = ap.parse_args()

    variants = [dict(check_every=ce, exit_every=0) for ce in (5, 10, 15, 20, 30, 60)] if a.ladder \
        else [dict(check_every=a.check_every, exit_every=a.exit_every)]
    common = dict(lookback=a.lookback, vol_target=a.vol_target, lev_cap=a.lev_cap, cost=a.cost_bp / 1e4, long_only=a.long_only)

    out = {}
    all_sr = []
    for sym in a.symbols:
        days = load_days(sym)
        dates = [d for d, *_ in days]
        out[sym] = {"span": [str(dates[0]), str(dates[-1])], "n_sessions": len(days), "variants": {}}
        for v in variants:
            r, ntr = run_strategy(days, **v, **common)
            s = stats(r, dates)
            s["trades_per_day"] = float(ntr.mean())
            s["params"] = {**v, **common}
            out[sym]["variants"][f"decide_{v['check_every']}m_exit_{v['exit_every']}m"] = s
            all_sr.append(s["sr_daily"])
    n_trials = a.trials or len(all_sr)
    var_sr = float(np.var([x for x in all_sr if x == x], ddof=1)) if len(all_sr) > 1 else 0.0
    for sym in out:
        for s in out[sym]["variants"].values():
            s["deflated_psr"] = deflated_sharpe(s["sr_daily"], s["n_days"], s["skew"], s["kurt"], n_trials, var_sr)
            s["n_trials"] = n_trials

    print(f"\nlev cap {a.lev_cap}x | cost {a.cost_bp} bp per round trip | fills at the next 1-min open | {n_trials} variants counted in the deflation")
    for sym, res in out.items():
        print(f"\n[{sym}] {res['n_sessions']} sessions {res['span'][0]} -> {res['span'][1]}")
        print(f"  {'variant':22} {'annRet':>7} {'annVol':>7} {'Sharpe':>7} {'t':>6} {'maxDD':>7} {'tr/day':>6} {'16-19':>6} {'20-26':>6} {'deflPSR':>8}")
        for name, s in res["variants"].items():
            print(f"  {name:22} {s['ann_return']:7.2%} {s['ann_vol']:7.2%} {s['sharpe']:7.2f} {s['t_stat']:6.2f} {s['max_drawdown']:7.1%} "
                  f"{s['trades_per_day']:6.2f} {s['sharpe_2016_2019']:6.2f} {s['sharpe_2020_2026']:6.2f} {s['deflated_psr']:8.2f}")
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / ("ladder.json" if a.ladder else "backtest.json")
    f.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {f}")
    if not a.ladder:
        pnl_report(a, common, variants[0])
    print("Reminder: every variant you try counts. Re-run with --trials <total tried> and watch the deflated PSR.")


# ------------------------------------------------------------------------ P&L view
START_EQUITY = 100_000.0


def pnl_report(a, common: dict, variant: dict) -> None:
    """Dollar P&L on $100k: equity curves per ETF and for the book the bot runs (equal weight of
    bot.SYMBOLS, i.e. 50% QQQ + 50% SPY), total and by year. Writes CSV, JSON and a chart."""
    try:
        from bot import SYMBOLS as BOT_SYMBOLS
    except Exception:
        BOT_SYMBOLS = ["QQQ", "SPY"]
    curves, rets = {}, {}
    for sym in a.symbols:
        days = load_days(sym)
        r, _ = run_strategy(days, **variant, **common)
        rets[sym] = pd.Series(r, index=pd.to_datetime([d for d, *_ in days]))
    book = [s for s in BOT_SYMBOLS if s in rets]
    if len(book) >= 2:
        rets["BOOK (" + "+".join(book) + ", equal weight)"] = pd.concat([rets[s] for s in book], axis=1).fillna(0.0).mean(axis=1)
    eq = pd.DataFrame({k: START_EQUITY * (1 + v).cumprod() for k, v in rets.items()}).ffill()   # union of the calendars
    pnl_by_year = {}
    for k, v in rets.items():
        s = START_EQUITY * (1 + v).cumprod()          # each series on its own calendar: no gaps to mis-fill
        daily_pnl = s.diff()
        daily_pnl.iloc[0] = s.iloc[0] - START_EQUITY
        pnl_by_year[k] = daily_pnl.groupby(s.index.year).sum()
    summary = {"start_equity": START_EQUITY, "lev_cap": a.lev_cap, "cost_bp": a.cost_bp, "variant": variant,
               "total_pnl": {k: float(eq[k].iloc[-1] - START_EQUITY) for k in eq},
               "final_equity": {k: float(eq[k].iloc[-1]) for k in eq},
               "max_drawdown": {k: float((eq[k] / eq[k].cummax() - 1).min()) for k in eq},
               "pnl_by_year": {k: {str(y): float(v) for y, v in s.items()} for k, s in pnl_by_year.items()}}
    tag = f"lev{a.lev_cap:g}x"
    eq.to_csv(RESULTS / f"equity_{tag}.csv", index_label="date")
    (RESULTS / f"pnl_{tag}.json").write_text(json.dumps(summary, indent=2))
    years = sorted({y for s in pnl_by_year.values() for y in s.index})
    print(f"\nP&L on ${START_EQUITY:,.0f} starting equity, {tag}, {a.cost_bp} bp per round trip")
    print(f"  {'':34}" + "".join(f"{y:>9}" for y in years) + f"{'total':>11}{'maxDD':>8}")
    for k in eq:
        row = "".join(f"{pnl_by_year[k].get(y, 0.0):>9,.0f}" for y in years)
        print(f"  {k:34}{row}{summary['total_pnl'][k]:>11,.0f}{summary['max_drawdown'][k]:>8.1%}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for k in eq:
            ax.plot(eq.index, eq[k], label=f"{k}  ({summary['total_pnl'][k]:+,.0f})", linewidth=2.2 if k.startswith("BOOK") else 1.3)
        ax.axhline(START_EQUITY, color="grey", linewidth=0.8)
        ax.set_title(f"Noise-area intraday momentum: equity on ${START_EQUITY:,.0f}, {tag}, {a.cost_bp} bp/round trip, flat every night")
        ax.set_ylabel("equity ($)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"pnl_{tag}.png", dpi=150)
        print(f"-> {RESULTS / f'pnl_{tag}.png'}, {RESULTS / f'equity_{tag}.csv'}")
    except Exception as e:                                      # matplotlib is optional
        print(f"(no chart: {e})")


if __name__ == "__main__":
    main()
