# SCF Alpaca Trading Bot Example

A day-trading bot for an Alpaca **paper** account, and the backtest that decides whether
its rule is worth running. Three Python files, no framework.

```
bot.py          the bot: run it in the morning, it trades QQQ and SPY through the day, flat by 15:55
backtest.py     the same rule on ten years of 1-minute bars, with costs and a deflated Sharpe
pull_data.py    downloads the 1-minute bars the backtest needs (free Alpaca plan is enough)
slides/         the six class slides
```

## Setup (15 minutes)

1. Python 3.11+ and the packages: `pip install -r requirements.txt`
2. A free Alpaca account at alpaca.markets. Open the **Paper** dashboard and generate an API key pair.
3. Copy `secrets.env.example` to `secrets.env` and paste the two keys in. That file is git-ignored.
   Never paste keys into code, chat, screenshots or commits. If you do, regenerate them.
4. Pull the data once (a few minutes per symbol):
   ```
   python pull_data.py
   ```

## Backtest

```
python backtest.py                      # QQQ SPY IWM, decide every 30 min, stop check every 5, 1 bp per round trip
python backtest.py --ladder             # decision clocks 5 / 10 / 15 / 20 / 30 / 60 minutes
python backtest.py --cost-bp 2          # what doubling costs does to it
python backtest.py --lev-cap 4          # the paper's leverage: same Sharpe, 4x the swings and drawdowns
```

What you should see (2016-01 → 2026-09, unlevered, 1 bp): QQQ Sharpe about 1.1 to 1.2 with a
t-stat near 4 in both halves of the decade; SPY about 0.6; IWM negative. The ladder shows every
faster clock losing edge: QQQ 1.22 at 30 minutes, 0.91 at 15, 0.58 at 5.

Every non-ladder run also prints the **P&L in dollars** on $100,000, by year, for each ETF and
for the book the bot runs (50% QQQ + 50% SPY), and writes `results/pnl_lev1x.png` (the equity
curve), `results/equity_lev1x.csv` and `results/pnl_lev1x.json`. At 1× the book turns $100,000
into about $184,000 over the decade with a 7% worst drawdown and three losing years; at
`--lev-cap 4` it is about $299,000 with a 17% drawdown. Leverage scales both.

The backtest is honest by construction: the band for day *t* uses only days *t−14…t−1*, the
signal is the last bar before the decision and the fill is the next bar's open, costs are charged on
every round trip, and the Sharpe is deflated for the number of variants you ran. Run `--ladder`
and you have tried six things: pass `--trials 6` next time and watch the deflated PSR fall.
That is the point.

## Run the bot

```
python bot.py                  # dry run: decides and logs, sends nothing
python bot.py --live-paper     # sends PAPER orders
python bot.py --lev-cap 4      # showcase mode: bigger movements on the paper account
python bot.py --help
```

Start it any time before the open (it waits), or during the day (it joins at the next decision).
What it does:

| ET | action |
|---|---|
| 09:30 | loads the last 14 sessions of 1-minute bars, builds today's band from the open and yesterday's close |
| 10:00 → 15:30, every 30 min | price above the band → long, below → short, inside → flat; one market order per change |
| every 5 min in between | if an open trade is back inside the band → flat (stop check only, never a new entry) |
| every minute | logs the mark: price, band, position → `logs/bot_YYYY-MM-DD.log` |
| 15:55 | flat. Nothing is held overnight |

Sizing: 50% of equity per symbol × min(2% ÷ 14-day daily vol, leverage cap). Settings are the
block at the top of `bot.py`; every one is also a command-line flag.

**Safety.** Orders go only to a host containing `paper-api`, and only with `--live-paper`. There is
no live-money code path, and you should not add one.

## The rule

Zarattini, Barbon & Aziz (2024), *Beat the Market: An Effective Intraday Momentum Strategy for
the S&P500 ETF*. Around max/min(today's open, yesterday's close) sits a band whose width at each
minute is the average absolute move from the open at that minute over the previous 14 days. Above
the band, momentum is real → long. Below → short. Inside → noise → flat. The band is the trailing
stop. Rebuilt from the paper and tested on our own data.

## Read it like a quant

- It works on one ETF, half-works on another, loses on the third. It took six failed ideas to find
  (first-half-hour momentum, trading every 30 minutes on the last sign, opening-range breakouts, a
  faster clock, a gap screen on single stocks). The deflated Sharpe on QQQ, after 16 variants, is 0.14.
- Costs matter: at 2 bp per round trip the 15-minute version halves. Frequency is a cost you pay,
  not an edge you gain, unless the signal lives at that frequency. This one doesn't.
- Two real paper sessions: +$42, then −$458 at 4× on a day QQQ hugged its band. Every rule fired
  correctly both days. That is what a Sharpe-1 intraday strategy looks like on any given Tuesday.
- Getting alpha out of the market over the long run is hard. If it were easy, none of us would be here.
