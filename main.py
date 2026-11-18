"""Dip-buying mean-reversion bot for btc-race-24h-20261118-1444.

Trades CBBTC/USDC on 1-minute bars, long only, at most one open position,
100% of equity per trade. Speaks agent-fun/bars-replay/v1 through the
vendored kit shim (vendor/makefun_bars.py, byte-for-byte copy of the kit SDK).

Strategy, chosen from a grid search over this competition's own 1440-bar dev
window (see tuning-notes.md in the work dir):
  ENTRY (flat only): the latest close sits at least DIP_BPS below its
    SMA_N close average, RSI(14) is oversold, and the close has bounced at
    least BOUNCE_BPS above the lowest low of the last BOUNCE_WIN bars.
    A dip that is still falling does not trigger; the bounce filter waits
    for the turn.
  EXITS (while holding): take-profit, stop-loss, a time stop, and a trailing
    stop that arms once the trade is up TRAIL_ARM_BPS and exits on a
    TRAIL_PB_BPS pullback from the peak close since entry.
  Orders fill at the next bar's open, so every decision is made on the close
  just received; max_slippage_bps is 1000, far above any plausible one-minute
  move plus the modelled impact of a full-size order on this pool.

All arithmetic is integer (fixed-point RSI, integer SMA), no randomness, no
wall clock, no environment, no files: two runs on the same bars produce the
same replies byte for byte.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- tuned parameters (dev window: net +$22.58, 1 round trip, max DD 1.50%) ---
SMA_N = 60            # bars in the dip reference average
DIP_BPS = 130         # entry: close this far below the SMA (bps)
RSI_N = 14            # Wilder RSI period
RSI_MAX = 40          # entry: RSI(14) below this (0-100 scale)
BOUNCE_BPS = 10       # entry: close this far above the recent low (bps)
BOUNCE_WIN = 10       # bars over which the recent low is taken
TP_BPS = 250          # exit: take profit (bps above entry fill price)
SL_BPS = 300          # exit: stop loss (bps below entry fill price)
TIME_STOP = 720       # exit: bars held before the time stop fires
TRAIL_ARM_BPS = 150   # trailing stop arms once up this far (bps)
TRAIL_PB_BPS = 75     # trailing stop exits on this pullback from peak (bps)
MAX_SLIP_BPS = 1000   # per-order slippage cap vs the latest close
# A trade needs time to reach a 250 bps take-profit on ~7 bps/bar noise, so
# entering in the final 120 bars of the 1440-bar window is negative-EV: the
# dev window's own late dip (bar 1436) flattened at a loss in the looser
# configs. No new entries once the batch seq passes this point.
NO_ENTRY_AFTER_SEQ = 1320

SCALE = 1000000       # fixed-point scale for the RSI averages
BPS = 10000


class Bot:
    """Per-run state. Everything here is a pure function of the bars seen."""

    def __init__(self):
        self.closes = []        # int closes, price_scale 5
        self.lows = []          # int lows
        self.prev_close = None
        self.gain_sum = 0       # sum of the first RSI_N gains (price units)
        self.loss_sum = 0       # sum of the first RSI_N losses (price units)
        self.diff_count = 0     # price diffs seen so far
        self.avg_gain = None    # Wilder avg gain, fixed point (x SCALE)
        self.avg_loss = None    # Wilder avg loss, fixed point (x SCALE)
        self.entry_price = None  # fill price of the open position
        self.entry_seq = None
        self.peak = None        # highest close since entry
        self.armed = False      # trailing stop armed

    def rsi(self):
        """Wilder RSI(14) x SCALE, or None before 14 diffs exist."""
        if self.avg_gain is None:
            return None
        total = self.avg_gain + self.avg_loss
        if total == 0:
            return 50 * SCALE
        return 100 * SCALE * self.avg_gain // total

    def note_bar(self, close, low):
        self.closes.append(close)
        self.lows.append(low)
        if self.prev_close is not None:
            diff = close - self.prev_close
            gain = diff if diff > 0 else 0
            loss = -diff if diff < 0 else 0
            self.diff_count += 1
            if self.diff_count <= RSI_N:
                self.gain_sum += gain
                self.loss_sum += loss
                if self.diff_count == RSI_N:
                    self.avg_gain = self.gain_sum * SCALE // RSI_N
                    self.avg_loss = self.loss_sum * SCALE // RSI_N
            else:
                self.avg_gain = (self.avg_gain * (RSI_N - 1) + gain * SCALE) // RSI_N
                self.avg_loss = (self.avg_loss * (RSI_N - 1) + loss * SCALE) // RSI_N
        self.prev_close = close

    def note_fills(self, fills):
        for fill in fills:
            if fill.get("status") != "filled":
                continue
            if fill.get("symbol") != SYMBOL:
                continue
            if fill.get("side") == "buy":
                self.entry_price = int(fill["price"])
                self.entry_seq = self.seq
                self.peak = self.closes[-1]
                self.armed = False
            elif fill.get("side") == "sell":
                self.entry_price = None
                self.entry_seq = None
                self.peak = None
                self.armed = False

    def on_batch(self, state, batch):
        self.seq = state.seq
        bar = state.latest.get(SYMBOL)
        if bar is None:
            return []
        close = int(bar["c"])
        low = int(bar["l"])
        self.note_bar(close, low)
        self.note_fills(batch.get("fills", []))

        holding = state.positions.get(SYMBOL)
        if holding is None or holding.qty == 0:
            return self.maybe_enter(state, close)
        return self.maybe_exit(state, close, holding.qty)

    def maybe_enter(self, state, close):
        if state.warming_up:
            return []
        if state.seq >= NO_ENTRY_AFTER_SEQ:
            return []
        if len(self.closes) < SMA_N:
            return []
        rsi = self.rsi()
        if rsi is None or rsi >= RSI_MAX * SCALE:
            return []
        sma = sum(self.closes[-SMA_N:]) // SMA_N
        if close * BPS >= sma * (BPS - DIP_BPS):
            return []
        recent_low = min(self.lows[-BOUNCE_WIN:])
        if close * BPS < recent_low * (BPS + BOUNCE_BPS):
            return []
        cash = state.cash
        if cash < 1000000:  # below the 1 USDC minimum notional
            return []
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": cash,
                 "max_slippage_bps": MAX_SLIP_BPS}]

    def maybe_exit(self, state, close, qty):
        if self.entry_price is None:
            # Position without a recorded fill price (should not happen):
            # mark from the current close so exits still work.
            self.entry_price = close
            self.entry_seq = state.seq
            self.peak = close
            self.armed = False
        if close > self.peak:
            self.peak = close
        if not self.armed and close * BPS >= self.entry_price * (BPS + TRAIL_ARM_BPS):
            self.armed = True
        sell = False
        if close * BPS >= self.entry_price * (BPS + TP_BPS):
            sell = True
        elif close * BPS <= self.entry_price * (BPS - SL_BPS):
            sell = True
        elif state.seq - self.entry_seq >= TIME_STOP:
            sell = True
        elif self.armed and close * BPS <= self.peak * (BPS - TRAIL_PB_BPS):
            sell = True
        if not sell:
            return []
        return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                 "max_slippage_bps": MAX_SLIP_BPS}]


def main():
    bot = Bot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
